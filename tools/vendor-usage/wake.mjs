// When a usage source is temporarily unreadable, wake the relevant management
// CLI once and retry. This never sends an inference request and never handles
// credentials itself.

import { execFile, spawn } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

export const WAKE_TIMEOUT_MS = 8_000;
export const WAKE_BUDGET_MS = 8_000;
export const RETRY_RESERVE_MS = 2_000;
const POLL_MS = 200;
const CONCURRENT_ATTEMPTS = 4;

// NoteFactory deliberately wakes only the two user-approved recovery paths.
// Claude is excluded: its useful refresh is an interactive no-argument launch,
// which is not an acceptable hidden side effect for this application.
export const WAKE_COMMANDS = Object.freeze({
  // AGY's process is the loopback RPC server. Re-read while `agy models` is
  // still alive, then kill it if it outlives the bounded retry window.
  antigravity: Object.freeze({
    file: "agy", args: Object.freeze(["models"]), mode: "concurrent", attempts: 4,
    reasons: Object.freeze(new Set(["local_rpc_unavailable"])),
  }),
  // Grok refreshes its own auth state before this management command returns.
  grok: Object.freeze({
    file: "grok", args: Object.freeze(["models"]), mode: "await",
    reasons: Object.freeze(new Set(["auth_error", "credential_missing"])),
  }),
});

export const WAKEABLE_REASONS = Object.freeze(new Set([
  "auth_error", "credential_missing", "local_rpc_unavailable",
]));

const delay = (ms) => new Promise((resolve) => { setTimeout(resolve, ms); });

function wakeInvocation(command, { platform = process.platform } = {}) {
  // Windows npm installations can expose Grok as a .cmd shim. Use explicit
  // cmd.exe with a frozen management command rather than shell:true.
  if (platform === "win32" && command.file === "grok") {
    return { file: "cmd.exe", args: ["/d", "/s", "/c", "grok models"] };
  }
  return { file: command.file, args: [...command.args] };
}

async function wakeThenFetch(command, fetch, args, {
  run = execFileAsync, timeoutMs = WAKE_TIMEOUT_MS, nowFn = Date.now, platform = process.platform,
} = {}) {
  const deadline = nowFn() + (command.budgetMs ?? WAKE_BUDGET_MS);
  try {
    const invocation = wakeInvocation(command, { platform });
    const commandTimeout = Math.max(0, Math.min(timeoutMs, deadline - nowFn() - RETRY_RESERVE_MS));
    if (commandTimeout <= 0) return null;
    await run(invocation.file, invocation.args, {
      windowsHide: true, timeout: commandTimeout, maxBuffer: 1024 * 1024,
    });
  } catch {
    return null;
  }
  const remaining = deadline - nowFn();
  return remaining > 0 ? fetch({ ...args, timeoutMs: remaining }) : null;
}

async function wakeWhileFetching(command, fetch, args, {
  spawnImpl = spawn, pollMs = POLL_MS, nowFn = Date.now, platform = process.platform,
} = {}) {
  const attempts = command.attempts ?? CONCURRENT_ATTEMPTS;
  const deadline = nowFn() + (command.budgetMs ?? WAKE_BUDGET_MS);
  let child;
  try {
    const invocation = wakeInvocation(command, { platform });
    child = spawnImpl(invocation.file, invocation.args, { windowsHide: true, stdio: "ignore" });
  } catch {
    return null;
  }
  let launchFailed = false;
  child.on?.("error", () => { launchFailed = true; });
  try {
    let last = null;
    for (let i = 0; i < attempts; i += 1) {
      await delay(pollMs);
      if (launchFailed) return null;
      const remaining = deadline - nowFn();
      if (remaining <= 0) return last;
      last = await fetch({ ...args, timeoutMs: remaining });
      if (last?.ok || launchFailed) return last;
      if (child.exitCode !== null || child.signalCode !== null) return last;
    }
    return last;
  } finally {
    if (child.exitCode === null && child.signalCode === null) {
      try { child.kill?.(); } catch { /* process already exited */ }
    }
  }
}

export function wrapFetcher(provider, fetch, options = {}) {
  const command = WAKE_COMMANDS[provider];
  if (!command) return fetch;
  return async (args) => {
    const first = await fetch(args);
    if (first?.ok || !command.reasons?.has(first?.reason)) return first;
    const retried = command.mode === "concurrent"
      ? await wakeWhileFetching(command, fetch, args, options)
      : await wakeThenFetch(command, fetch, args, options);
    return retried ?? first;
  };
}

// Production default fetchers are wrapped. Injected fetchers stay untouched so
// deterministic tests cannot spawn a vendor process.
export function withWake(fetchers, options = {}) {
  return Object.freeze(Object.fromEntries(
    Object.entries(fetchers).map(([provider, fetch]) => [provider, wrapFetcher(provider, fetch, options)]),
  ));
}
