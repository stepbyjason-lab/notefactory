// Claude OAuth usage adapter. Read-only credential discovery; no refresh or write path.

import { createHash } from "node:crypto";
import { execFile } from "node:child_process";
import { readFile } from "node:fs/promises";
import { homedir } from "node:os";
import path from "node:path";
import { promisify } from "node:util";
import { failure, normalizeUsedPercentWindow, requestJson, runWithDeadline } from "./source-utils.mjs";

const PROVIDER = "claude";
const SOURCE = "claude-oauth";
const ORIGIN = "https://api.anthropic.com";
const URL = `${ORIGIN}/api/oauth/usage`;
const SERVICE = "Claude Code-credentials";
const execFileAsync = promisify(execFile);

export function deriveClaudeKeychainIdentity({ configDir, env = {} } = {}) {
  const account = env.USER || env.USERNAME || "user";
  const service = configDir
    ? `${SERVICE}-${createHash("sha256").update(configDir).digest("hex").slice(0, 8)}`
    : SERVICE;
  return { service, account };
}

function credentialFromText(text) {
  const oauth = JSON.parse(text)?.claudeAiOauth;
  return {
    accessToken: typeof oauth?.accessToken === "string" && oauth.accessToken.trim() ? oauth.accessToken : null,
    refreshOnly: typeof oauth?.refreshToken === "string" && oauth.refreshToken.trim().length > 0,
  };
}

async function readKeychainCandidate(keychainLookup, configDir, env, signal) {
  if (typeof keychainLookup !== "function") return { accessToken: null, refreshOnly: false };
  try {
    const identity = deriveClaudeKeychainIdentity({ configDir, env });
    return credentialFromText(await keychainLookup({ ...identity, signal }));
  } catch {
    return { accessToken: null, refreshOnly: false };
  }
}

async function defaultKeychainLookup({ service, account, signal }) {
  if (process.platform !== "darwin") return null;
  try {
    const { stdout } = await execFileAsync("security", ["find-generic-password", "-s", service, "-a", account, "-w"], { timeout: 3_000, maxBuffer: 256 * 1024, signal });
    return stdout.trim() || null;
  } catch {
    return null;
  }
}

export async function readClaudeCredential({ configDir, platform = process.platform, env = process.env, keychainLookup, readText = readFile, signal } = {}) {
  if (platform === "darwin") {
    const lookup = keychainLookup ?? defaultKeychainLookup;
    if (configDir) {
      const scoped = await readKeychainCandidate(lookup, configDir, env, signal);
      const legacy = await readKeychainCandidate(lookup, undefined, env, signal);
      if (scoped.accessToken) return scoped;
      if (legacy.accessToken) return legacy;
      if (scoped.refreshOnly || legacy.refreshOnly) return { accessToken: null, refreshOnly: true };
    } else {
      const legacy = await readKeychainCandidate(lookup, undefined, env, signal);
      if (legacy.accessToken || legacy.refreshOnly) return legacy;
    }
  }
  const directory = configDir ?? path.join(homedir(), ".claude");
  try {
    return credentialFromText(await readText(path.join(directory, ".credentials.json"), "utf8"));
  } catch (error) {
    if (error instanceof SyntaxError) return { accessToken: null, refreshOnly: false, reason: "parse_error" };
    if (error?.code === "ENOENT") return { accessToken: null, refreshOnly: false, reason: "credential_missing" };
    return { accessToken: null, refreshOnly: false, reason: "source_error" };
  }
}

function normalizeClaudePayload(payload, nowMs) {
  if (!payload?.five_hour || typeof payload.five_hour !== "object" || !payload?.seven_day || typeof payload.seven_day !== "object") {
    return { reason: "window_unclassifiable" };
  }
  const five = normalizeUsedPercentWindow({ usedPercent: payload?.five_hour?.utilization ?? payload?.five_hour?.used_percentage, resetTime: payload?.five_hour?.resets_at, windowMinutes: 300, sourceWindow: "5h", nowMs });
  const seven = normalizeUsedPercentWindow({ usedPercent: payload?.seven_day?.utilization ?? payload?.seven_day?.used_percentage, resetTime: payload?.seven_day?.resets_at, windowMinutes: 10080, sourceWindow: "weekly", nowMs });
  return five.window && seven.window ? { windows: { five: five.window, seven: seven.window } } : { reason: five.reason ?? seven.reason ?? "window_unclassifiable" };
}

export async function fetchClaudeUsage(options = {}) {
  const outcome = await runWithDeadline({
    timeoutMs: options.timeoutMs,
    run: async (deadline) => {
      const credential = await readClaudeCredential({ ...options, signal: deadline.signal });
      if (credential.reason) return failure(PROVIDER, SOURCE, credential.reason, { target_origin: ORIGIN });
      if (!credential.accessToken) return failure(PROVIDER, SOURCE, "credential_missing", { target_origin: ORIGIN });
      const result = await requestJson({
        request: options.request,
        url: URL,
        provider: PROVIDER,
        source: SOURCE,
        targetOrigin: ORIGIN,
        timeoutMs: options.timeoutMs,
        deadline,
        init: { headers: { Authorization: `Bearer ${credential.accessToken}`, "anthropic-beta": "oauth-2025-04-20", "User-Agent": "claude-code/2.1.0" } },
      });
      if (!result.ok) return result;
      const normalized = normalizeClaudePayload(result.value, options.nowMs ?? Date.now());
      return normalized.windows ? { ok: true, provider: PROVIDER, source: SOURCE, windows: normalized.windows } : failure(PROVIDER, SOURCE, normalized.reason, { target_origin: ORIGIN });
    },
  });
  if (outcome.timedOut) return failure(PROVIDER, SOURCE, "source_timeout", { target_origin: ORIGIN });
  if (outcome.error) return failure(PROVIDER, SOURCE, "source_error", { target_origin: ORIGIN });
  return outcome.value;
}
