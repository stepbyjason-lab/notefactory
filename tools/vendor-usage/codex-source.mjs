// Codex OAuth usage adapter. It reads only the selected auth.json file.

import { readFile } from "node:fs/promises";
import { homedir } from "node:os";
import path from "node:path";
import { failure, normalizeUsedPercentWindow, requestJson, runWithDeadline } from "./source-utils.mjs";

const PROVIDER = "codex";
const SOURCE = "codex-oauth";
const ORIGIN = "https://chatgpt.com";
const URL = `${ORIGIN}/backend-api/wham/usage`;

export function resolveCodexAuthPath({ codexHomePath, env = process.env, homeDir = homedir() } = {}) {
  return path.join(codexHomePath ?? env.CODEX_HOME ?? path.join(homeDir, ".codex"), "auth.json");
}

async function readCodexCredential(options) {
  try {
    const content = await (options.readText ?? readFile)(resolveCodexAuthPath(options), "utf8");
    const tokens = JSON.parse(content)?.tokens;
    if (typeof tokens?.access_token !== "string" || !tokens.access_token.trim()) return { missing: true };
    return { accessToken: tokens.access_token, accountId: typeof tokens.account_id === "string" ? tokens.account_id : null };
  } catch (error) {
    if (error instanceof SyntaxError) return { parseError: true };
    if (error?.code === "ENOENT") return { missing: true };
    return { sourceError: true };
  }
}

function normalizeCodexPayload(payload, nowMs) {
  const rateLimit = payload.rate_limit;
  const windows = { five: null, seven: null };
  const reasons = [];
  const definitions = {
    18000: { key: "five", windowMinutes: 300, sourceWindow: "5h" },
    604800: { key: "seven", windowMinutes: 10080, sourceWindow: "weekly" },
  };

  // Primary wins ties: scan order is fixed and a later duplicate never replaces
  // an already-normalized period.
  for (const rawWindow of [rateLimit?.primary_window, rateLimit?.secondary_window]) {
    const definition = definitions[rawWindow?.limit_window_seconds];
    if (!definition) continue;
    const normalized = normalizeUsedPercentWindow({
      usedPercent: rawWindow.used_percent,
      resetTime: rawWindow.reset_at,
      windowMinutes: definition.windowMinutes,
      sourceWindow: definition.sourceWindow,
      nowMs,
    });
    if (normalized.window) {
      if (!windows[definition.key]) windows[definition.key] = normalized.window;
    } else {
      reasons.push(normalized.reason);
    }
  }

  return windows.five || windows.seven ? { windows } : { reason: reasons[0] ?? "window_unclassifiable" };
}

export async function fetchCodexUsage(options = {}) {
  const outcome = await runWithDeadline({
    timeoutMs: options.timeoutMs,
    run: async (deadline) => {
      const credential = await readCodexCredential(options);
      if (credential.parseError) return failure(PROVIDER, SOURCE, "parse_error", { target_origin: ORIGIN });
      if (credential.sourceError) return failure(PROVIDER, SOURCE, "source_error", { target_origin: ORIGIN });
      if (credential.missing) return failure(PROVIDER, SOURCE, "credential_missing", { target_origin: ORIGIN });
      const headers = { Authorization: `Bearer ${credential.accessToken}`, "User-Agent": "codex-cli", "OpenAI-Beta": "codex-1", originator: "Codex Desktop" };
      if (credential.accountId) headers["ChatGPT-Account-Id"] = credential.accountId;
      const result = await requestJson({ request: options.request, url: URL, provider: PROVIDER, source: SOURCE, targetOrigin: ORIGIN, timeoutMs: options.timeoutMs, deadline, init: { headers } });
      if (!result.ok) return result;
      const normalized = normalizeCodexPayload(result.value, options.nowMs ?? Date.now());
      return normalized.windows ? { ok: true, provider: PROVIDER, source: SOURCE, windows: normalized.windows } : failure(PROVIDER, SOURCE, normalized.reason, { target_origin: ORIGIN });
    },
  });
  if (outcome.timedOut) return failure(PROVIDER, SOURCE, "source_timeout", { target_origin: ORIGIN });
  if (outcome.error) return failure(PROVIDER, SOURCE, "source_error", { target_origin: ORIGIN });
  return outcome.value;
}
