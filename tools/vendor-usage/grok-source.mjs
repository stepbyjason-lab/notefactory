// Grok OAuth usage adapter. It reads only the Grok CLI auth.json file.

import { readFile } from "node:fs/promises";
import { homedir } from "node:os";
import path from "node:path";
import { failure, normalizeUsedPercentWindow, requestJson, runWithDeadline } from "./source-utils.mjs";

const PROVIDER = "grok";
const SOURCE = "grok-oauth";
const DEFAULT_BILLING_BASE = "https://cli-chat-proxy.grok.com/v1";
const PREFERRED_ISSUER = "https://auth.x.ai";
const WEEKLY_WINDOW_MINUTES = 10_080;
const TOKEN_SKEW_MS = 5 * 60_000;

export function resolveGrokAuthPath({ grokHomePath, env = process.env, homeDir = homedir() } = {}) {
  return path.join(grokHomePath ?? env.GROK_HOME ?? path.join(homeDir, ".grok"), "auth.json");
}

export function resolveGrokBillingBase({ env = process.env } = {}) {
  const configured = typeof env.GROK_CLI_CHAT_PROXY_BASE_URL === "string" ? env.GROK_CLI_CHAT_PROXY_BASE_URL.trim() : "";
  return (configured || DEFAULT_BILLING_BASE).replace(/\/+$/, "");
}

function isPreferredIssuer(issuer) {
  return issuer === PREFERRED_ISSUER || issuer.startsWith(`${PREFERRED_ISSUER}::`);
}

function parseExpiry(value) {
  const parsed = typeof value === "string" ? Date.parse(value) : Number.NaN;
  return Number.isFinite(parsed) ? parsed : null;
}

function credentialFromEntry(entry) {
  if (!entry || typeof entry !== "object" || typeof entry.key !== "string" || !entry.key.trim()) return null;
  return {
    accessToken: entry.key,
    userId: typeof entry.user_id === "string" && entry.user_id.trim() ? entry.user_id : null,
    expiresAtMs: parseExpiry(entry.expires_at),
  };
}

export async function readGrokCredential(options = {}) {
  try {
    const content = await (options.readText ?? readFile)(resolveGrokAuthPath(options), "utf8");
    const auth = JSON.parse(content);
    if (!auth || typeof auth !== "object" || Array.isArray(auth)) return { parseError: true };

    let preferredIssuerSeen = false;
    let expiredPreferred = null;
    let fallback = null;
    for (const [issuer, entry] of Object.entries(auth)) {
      const preferred = isPreferredIssuer(issuer);
      preferredIssuerSeen ||= preferred;
      const credential = credentialFromEntry(entry);
      if (!credential) continue;
      if (preferred) {
        if (credential.expiresAtMs === null || credential.expiresAtMs - (options.nowMs ?? Date.now()) > TOKEN_SKEW_MS) return credential;
        expiredPreferred ??= credential;
      } else {
        fallback ??= credential;
      }
    }
    const selected = expiredPreferred ?? (preferredIssuerSeen ? null : fallback);
    if (!selected) return { missing: true };
    return { ...selected, expired: selected.expiresAtMs !== null && selected.expiresAtMs - (options.nowMs ?? Date.now()) <= TOKEN_SKEW_MS };
  } catch (error) {
    if (error instanceof SyntaxError) return { parseError: true };
    if (error?.code === "ENOENT") return { missing: true };
    return { sourceError: true };
  }
}

function timestampsMatch(left, right) {
  const leftMs = typeof left === "string" ? Date.parse(left) : Number.NaN;
  const rightMs = typeof right === "string" ? Date.parse(right) : Number.NaN;
  return Number.isFinite(leftMs) && leftMs === rightMs;
}

function hasConfirmedWeeklyPeriod(config) {
  const period = config?.currentPeriod;
  return period?.type === "USAGE_PERIOD_TYPE_WEEKLY"
    && timestampsMatch(period.start, config.billingPeriodStart)
    && timestampsMatch(period.end, config.billingPeriodEnd);
}

function billingConfig(payload) {
  if (payload?.config && typeof payload.config === "object") return payload.config;
  return payload && typeof payload === "object" ? payload : null;
}

function normalizeWeeklyCredits(config, nowMs) {
  const usedPercent = config?.creditUsagePercent === undefined && hasConfirmedWeeklyPeriod(config) ? 0 : config?.creditUsagePercent;
  const resetTime = config?.currentPeriod?.end ?? config?.billingPeriodEnd;
  return normalizeUsedPercentWindow({ usedPercent, resetTime, windowMinutes: WEEKLY_WINDOW_MINUTES, sourceWindow: "weekly", nowMs });
}

function hasMonthlyUsage(config) {
  const limit = Number(config?.monthlyLimit?.val);
  const used = Number(config?.used?.val);
  return Number.isFinite(limit) && limit > 0 && Number.isFinite(used);
}

export async function fetchGrokUsage(options = {}) {
  const outcome = await runWithDeadline({
    timeoutMs: options.timeoutMs,
    run: async (deadline) => {
      const base = resolveGrokBillingBase(options);
      let origin;
      try {
        const parsed = new URL(base);
        if (parsed.protocol !== "https:" || parsed.username || parsed.password) return failure(PROVIDER, SOURCE, "source_error");
        origin = parsed.origin;
      } catch { return failure(PROVIDER, SOURCE, "source_error"); }

      const credential = await readGrokCredential({ ...options, nowMs: options.nowMs ?? Date.now() });
      if (credential.parseError) return failure(PROVIDER, SOURCE, "parse_error", { target_origin: origin });
      if (credential.sourceError) return failure(PROVIDER, SOURCE, "source_error", { target_origin: origin });
      if (credential.missing) return failure(PROVIDER, SOURCE, "credential_missing", { target_origin: origin });
      if (credential.expired) return failure(PROVIDER, SOURCE, "auth_error", { target_origin: origin });

      const headers = { Authorization: `Bearer ${credential.accessToken}`, "X-XAI-Token-Auth": "xai-grok-cli", Accept: "application/json" };
      if (credential.userId) headers["x-userid"] = credential.userId;
      const request = (url) => requestJson({ request: options.request, url, provider: PROVIDER, source: SOURCE, targetOrigin: origin, timeoutMs: options.timeoutMs, deadline, init: { headers } });

      const credits = await request(`${base}/billing?format=credits`);
      if (!credits.ok) return credits;
      const creditConfig = billingConfig(credits.value);
      const weekly = normalizeWeeklyCredits(creditConfig, options.nowMs ?? Date.now());
      if (weekly.window) return { ok: true, provider: PROVIDER, source: SOURCE, windows: { five: null, seven: weekly.window } };

      const defaultBilling = await request(`${base}/billing`);
      if (!defaultBilling.ok) return defaultBilling;
      if (hasMonthlyUsage(billingConfig(defaultBilling.value))) return failure(PROVIDER, SOURCE, "window_unclassifiable", { target_origin: origin });
      return failure(PROVIDER, SOURCE, weekly.reason ?? "window_unclassifiable", { target_origin: origin });
    },
  });
  if (outcome.timedOut) return failure(PROVIDER, SOURCE, "source_timeout");
  if (outcome.error) return failure(PROVIDER, SOURCE, "source_error");
  return outcome.value;
}
