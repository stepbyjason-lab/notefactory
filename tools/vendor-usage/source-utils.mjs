// Shared bounded-I/O and quota-normalization helpers for R72a source adapters.
// These helpers never return credentials, raw HTTP bodies, headers, or command lines.

// NoteFactory is standalone: usage observations are returned to its caller,
// not written to Madi's project ledger or any shared runtime state.
function logCall() {}

export const SOURCE_TIMEOUT_MS = 10_000;
export const BODY_CAP_BYTES = 256 * 1024;

export const REASONS = Object.freeze(new Set([
  "credential_missing", "auth_error", "rate_limited", "redirect_disallowed",
  "source_timeout", "source_error", "parse_error", "invalid_reset", "stale_reset",
  "window_unclassifiable", "unsupported_platform", "unsupported_version",
  "local_rpc_unavailable", "process_ambiguous",
]));

const DETAIL_KEYS = new Set(["provider", "source", "reason", "http_status", "error_code", "target_origin"]);

// Raw diagnostics are intentionally discarded. Adapter callers may supply only
// the closed fields below; credentials, header values, bodies, paths, and
// command lines have no representable diagnostic slot.
export function redactDiagnostic(input = {}) {
  const safe = {};
  for (const [key, value] of Object.entries(input)) {
    if (DETAIL_KEYS.has(key) && value !== undefined) safe[key] = value;
  }
  return safe;
}

export function failure(provider, source, reason, detail = {}) {
  if (!REASONS.has(reason)) throw new TypeError(`unsupported source reason: ${reason}`);
  const safe = redactDiagnostic({ provider, source, reason, ...detail });
  return { ok: false, provider, source, reason, detail: safe };
}

function classifyHttpStatus(status) {
  if (status >= 300 && status < 400) return "redirect_disallowed";
  if (status === 401 || status === 403) return "auth_error";
  if (status === 429) return "rate_limited";
  return "source_error";
}

export function createDeadline(timeoutMs = SOURCE_TIMEOUT_MS, parentSignal) {
  const boundedTimeoutMs = Number.isFinite(timeoutMs) ? Math.max(0, Math.min(timeoutMs, SOURCE_TIMEOUT_MS)) : SOURCE_TIMEOUT_MS;
  const controller = new AbortController();
  const startedAt = Date.now();
  let timedOut = false;
  let timer;
  let expire;
  const expired = new Promise((resolve) => {
    expire = resolve;
    timer = setTimeout(() => {
      timedOut = true;
      controller.abort();
      resolve({ timedOut: true });
    }, boundedTimeoutMs);
  });
  const abortFromParent = () => {
    timedOut = true;
    controller.abort();
    expire({ timedOut: true });
  };
  if (parentSignal?.aborted) abortFromParent();
  else parentSignal?.addEventListener("abort", abortFromParent, { once: true });
  return {
    signal: controller.signal,
    get timedOut() { return timedOut; },
    remainingMs: () => Math.max(0, boundedTimeoutMs - (Date.now() - startedAt)),
    race: (promise) => Promise.race([Promise.resolve(promise).then((value) => ({ value }), (error) => ({ error })), expired]),
    close: () => {
      clearTimeout(timer);
      parentSignal?.removeEventListener("abort", abortFromParent);
    },
  };
}

export async function runWithDeadline({ timeoutMs = SOURCE_TIMEOUT_MS, parentSignal, run }) {
  const deadline = createDeadline(timeoutMs, parentSignal);
  try {
    if (deadline.signal.aborted) return { timedOut: true };
    return await deadline.race(Promise.resolve().then(() => run(deadline)));
  } finally {
    deadline.close();
  }
}

async function readCappedBody(response, deadline) {
  if (!response.body || typeof response.body.getReader !== "function") return { error: "source_error" };
  const reader = response.body.getReader();
  const chunks = [];
  let size = 0;
  try {
    for (;;) {
      const outcome = await deadline.race(reader.read());
      if (outcome.timedOut) {
        void reader.cancel().catch(() => {});
        return { error: "source_timeout" };
      }
      if (outcome.error) return { error: "source_error" };
      const { done, value } = outcome.value;
      if (done) break;
      size += value.byteLength;
      if (size > BODY_CAP_BYTES) {
        await reader.cancel();
        return { error: "source_error" };
      }
      chunks.push(value);
    }
  } catch {
    return { error: "source_error" };
  }
  return { text: new TextDecoder().decode(Buffer.concat(chunks)) };
}

// Bounded request primitive. `redirect:"manual"` makes every 3xx observable and
// rejects the common fetch default of following even same-origin redirects.
// `callContext`(entry·sessionId·turnId)는 호출 기록용이며 요청 자체에는 영향이 없다.
// 없으면 로그에 `unknown`으로 남고 조회는 그대로 돈다 — 옛 호출부를 안 고쳐도 된다.
export async function requestJson({ request = fetch, url, init, provider, source, targetOrigin, timeoutMs = SOURCE_TIMEOUT_MS, deadline, callContext }) {
  const ownedDeadline = deadline ?? createDeadline(timeoutMs);
  const note = (fields) => logCall({ provider, source, url, ...(callContext ?? {}), ...fields });
  try {
    if (ownedDeadline.signal.aborted) {
      note({ ok: false, reason: "source_timeout" });
      return failure(provider, source, "source_timeout", { target_origin: targetOrigin });
    }
    const outcome = await ownedDeadline.race(Promise.resolve().then(() => request(url, { ...init, redirect: "manual", signal: ownedDeadline.signal })));
    if (outcome.timedOut || ownedDeadline.signal.aborted) {
      note({ ok: false, reason: "source_timeout" });
      return failure(provider, source, "source_timeout", { target_origin: targetOrigin });
    }
    if (outcome.error) {
      note({ ok: false, reason: "source_error" });
      return failure(provider, source, "source_error", { target_origin: targetOrigin });
    }

    const response = outcome.value;
    const status = Number(response?.status);
    if (!Number.isInteger(status)) {
      note({ ok: false, reason: "source_error" });
      return failure(provider, source, "source_error", { target_origin: targetOrigin });
    }
    if (status < 200 || status >= 300) {
      // 실패일 때만 서버가 준 단서를 남긴다. `Retry-After` 가 없으면 429 의 창 길이를
      // 영영 모르고, 그래서 원인 판정이 매번 추측이 됐다(2026-08-28).
      let retryAfter = null;
      try { retryAfter = response?.headers?.get?.("retry-after") ?? null; } catch { /* 헤더 없음 */ }
      let bodyHint = null;
      try { bodyHint = (await readCappedBody(response, ownedDeadline))?.text ?? null; } catch { /* 본문 없음 */ }
      note({ ok: false, reason: classifyHttpStatus(status), httpStatus: status, retryAfter, bodyHint });
      return failure(provider, source, classifyHttpStatus(status), { http_status: status, target_origin: targetOrigin });
    }
    const body = await readCappedBody(response, ownedDeadline);
    if (body.error) {
      note({ ok: false, reason: body.error, httpStatus: status });
      return failure(provider, source, body.error, { target_origin: targetOrigin });
    }
    try {
      const value = JSON.parse(body.text);
      note({ ok: true, httpStatus: status });
      return { ok: true, value };
    } catch {
      note({ ok: false, reason: "parse_error", httpStatus: status });
      return failure(provider, source, "parse_error", { target_origin: targetOrigin });
    }
  } finally {
    if (!deadline) ownedDeadline.close();
  }
}

function parseReset(value) {
  if (typeof value === "number" && Number.isFinite(value)) return value < 10_000_000_000 ? value * 1000 : value;
  if (typeof value !== "string" || value.length === 0) return null;
  const asNumber = Number(value);
  if (Number.isFinite(asNumber) && value.trim() !== "") return asNumber < 10_000_000_000 ? asNumber * 1000 : asNumber;
  if (!/(Z|[+-]\d{2}:?\d{2})$/i.test(value)) return null;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function normalizeWindow({ remainingFraction, resetTime, windowMinutes, sourceWindow, nowMs }) {
  if (typeof remainingFraction !== "number" || !Number.isFinite(remainingFraction) || remainingFraction < 0 || remainingFraction > 1) {
    return { reason: "parse_error" };
  }
  const resetMs = parseReset(resetTime);
  if (resetMs === null) return { reason: "invalid_reset" };
  if (resetMs <= nowMs) return { reason: "stale_reset" };
  const minutesToReset = (resetMs - nowMs) / 60_000;
  return {
    window: {
      left: remainingFraction,
      remainingFraction,
      windowMinutes,
      sourceWindow,
      resetAt: new Date(resetMs).toISOString(),
      minutesToReset,
      timeLeftRatio: Math.min(1, minutesToReset / windowMinutes),
    },
  };
}

export function normalizeUsedPercentWindow({ usedPercent, resetTime, windowMinutes, sourceWindow, nowMs }) {
  if (typeof usedPercent !== "number" || !Number.isFinite(usedPercent) || usedPercent < 0 || usedPercent > 100) {
    return { reason: "parse_error" };
  }
  return normalizeWindow({ remainingFraction: (100 - usedPercent) / 100, resetTime, windowMinutes, sourceWindow, nowMs });
}
