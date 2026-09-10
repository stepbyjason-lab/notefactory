#!/usr/bin/env node
// Snapshot the three direct usage sources.  This is observational only: every
// provider is independent and this CLI always exits zero.

import process from "node:process";
import { pathToFileURL } from "node:url";
import { fetchClaudeUsage } from "./claude-source.mjs";
import { fetchCodexUsage } from "./codex-source.mjs";
import { fetchGrokUsage } from "./grok-source.mjs";
import { fetchAgyUsage } from "./agy-source.mjs";
import { withWake } from "./wake.mjs";
import { AGY_GROUP_NAMES } from "./agy-groups.mjs";

// Only the production default is wrapped. Explicitly injected fetchers are a
// deterministic test seam and must never start a vendor CLI.
const DEFAULT_FETCHERS = withWake({
  claude: fetchClaudeUsage,
  codex: fetchCodexUsage,
  grok: fetchGrokUsage,
  antigravity: fetchAgyUsage,
});

// Per-provider payload shape is deliberately the same as checkpoint's:
// `windows` stays a named key and `ok`/`provider`/`source` are present on both
// success and failure.  The only intended difference from checkpoint is that
// this CLI adds no verdict — no `decision`, no `status`.
//
// AGY group availability is surfaced as `availability`, not `status`, because
// `status` means a verdict in checkpoint.  Reusing the key for "is the
// measurement usable" made two sibling CLIs disagree on one word.

// Every emitted object carries the same key set whether it succeeded or not —
// absent values are explicit nulls.  A consumer can read `entry.windows.five`
// or `group.reason` without first branching on success.
function normalizeGroup(displayName, group) {
  const availability = group?.status ?? "unavailable";
  return {
    displayName: group?.displayName ?? displayName,
    ok: availability !== "unavailable",
    availability,
    windows: { five: group?.five ?? null, seven: group?.seven ?? null },
    reason: group?.reason ?? null,
  };
}

function unavailableGroups(reason) {
  return Object.fromEntries(Object.entries(AGY_GROUP_NAMES).map(([id, displayName]) => [id,
    normalizeGroup(displayName, { status: "unavailable", ...(reason ? { reason } : {}) }),
  ]));
}

function providerFailure(result, fallback = "source_error") {
  return {
    ok: false,
    provider: result?.provider ?? null,
    source: result?.source ?? null,
    reason: result?.reason ?? fallback,
    ...(result?.detail ? { detail: result.detail } : {}),
  };
}

function singleFailure(result, fallback) {
  return { ...providerFailure(result, fallback), windows: { five: null, seven: null } };
}

async function snapshotSingle(provider, fetcher, nowMs) {
  try {
    const result = await fetcher({ nowMs });
    if (!result?.ok) return singleFailure({ ...result, provider });
    return {
      ok: true,
      provider,
      source: result.source ?? null,
      reason: null,
      windows: { five: result.windows?.five ?? null, seven: result.windows?.seven ?? null },
    };
  } catch {
    return singleFailure({ provider, reason: "source_error" });
  }
}

async function snapshotAgy(fetcher, nowMs) {
  const provider = "antigravity";
  try {
    const result = await fetcher({ nowMs });
    if (result?.ok) {
      const groups = Object.fromEntries(Object.entries(AGY_GROUP_NAMES).map(([id, displayName]) => [id,
        normalizeGroup(displayName, result.groups?.[id]),
      ]));
      return { ok: true, provider, source: result.source ?? null, reason: null, groups };
    }
    const reason = result?.reason ?? "source_error";
    return { ...providerFailure({ ...result, provider, reason }), groups: unavailableGroups(reason) };
  } catch {
    return { ...providerFailure({ provider, reason: "source_error" }), groups: unavailableGroups("source_error") };
  }
}

export async function runSnapshot({
  nowMs = Date.now(),
  fetchers = DEFAULT_FETCHERS,
  // Transitional test seam for the former single-source caller.  Production
  // uses the explicit source map above.
  fetchImpl,
} = {}) {
  const get = (provider) => fetchImpl
    ? (options) => fetchImpl(provider, options)
    : fetchers[provider];
  const [claude, codex, grok, antigravity] = await Promise.all([
    snapshotSingle("claude", get("claude"), nowMs),
    snapshotSingle("codex", get("codex"), nowMs),
    snapshotSingle("grok", get("grok"), nowMs),
    snapshotAgy(get("antigravity"), nowMs),
  ]);
  return {
    ts: new Date(nowMs).toISOString(),
    providers: { claude, codex, grok, antigravity },
  };
}

function emit(envelope) {
  process.exitCode = 0;
  process.stdout.on("error", () => { process.exitCode = 0; });
  process.stdout.write(`${JSON.stringify(envelope)}\n`);
}

const isDirectRun = process.argv[1] && pathToFileURL(process.argv[1]).href.toLowerCase() === import.meta.url.toLowerCase();
if (isDirectRun) runSnapshot().then(emit).catch(() => emit({ ts: new Date().toISOString(), providers: {
  claude: providerFailure({ provider: "claude", reason: "source_error" }),
  codex: providerFailure({ provider: "codex", reason: "source_error" }),
  grok: providerFailure({ provider: "grok", reason: "source_error" }),
  antigravity: { ...providerFailure({ provider: "antigravity", reason: "source_error" }), groups: unavailableGroups("source_error") },
} }));
