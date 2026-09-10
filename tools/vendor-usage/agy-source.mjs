// Antigravity local-RPC adapter. Discovery is verify-before-probe: an
// unverified candidate never contributes a port or receives an HTTP request.

import { readFile } from "node:fs/promises";
import { execFile } from "node:child_process";
import { userInfo } from "node:os";
import { basename, dirname, join, normalize } from "node:path/win32";
import { promisify } from "node:util";
import { createDeadline, failure, normalizeWindow, requestJson, runWithDeadline } from "./source-utils.mjs";
import { AGY_GROUPS } from "./agy-groups.mjs";

const PROVIDER = "antigravity";
const SOURCE = "agy-local-rpc";
const ORIGIN = "127.0.0.1";
const RPC_PATH = "/exa.language_server_pb.LanguageServerService/RetrieveUserQuotaSummary";
const EXTENSION_IDENTITY = Object.freeze({ name: "antigravity", publisher: "google" });
// The CLI variant has no extension manifest to inspect. Its identity is the
// canonical per-user install path plus a valid Google LLC Authenticode CN.
const CLI_SIGNER_COMMON_NAME = "google llc";
const DISCOVERY_TIMEOUT_MS = 5_000;
const execFileAsync = promisify(execFile);

function flagValue(commandLine, name) {
  if (typeof commandLine !== "string") return null;
  const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = commandLine.match(new RegExp(`(?:^|\\s)${escaped}(?:=|\\s+)(?:\"([^\"]*)\"|'([^']*)'|([^\\s]+))`));
  return match ? (match[1] ?? match[2] ?? match[3] ?? null) : null;
}

function canonicalLanguageServerPath(executablePath, homeDir) {
  const normalized = normalize(executablePath).replaceAll("/", "\\");
  const expectedBin = normalize(join(homeDir, "AppData", "Local", "Programs", "Antigravity IDE", "resources", "app", "extensions", "antigravity", "bin"));
  return dirname(normalized).toLowerCase() === expectedBin.toLowerCase()
      && /^language_server(?:_[a-z0-9]+)*\.exe$/i.test(basename(normalized))
    ? normalized
    : null;
}

function canonicalAgyCliPath(executablePath, homeDir) {
  const normalized = normalize(executablePath).replaceAll("/", "\\");
  const expectedBin = normalize(join(homeDir, "AppData", "Local", "agy", "bin"));
  return dirname(normalized).toLowerCase() === expectedBin.toLowerCase() && isAgyCliName(normalized)
    ? normalized
    : null;
}

function isAgyCliName(executablePath) {
  return /^agy\.exe$/i.test(basename(normalize(executablePath).replaceAll("/", "\\")));
}

function isSignedByCliVendor(candidate) {
  const status = candidate?.signatureStatus ?? candidate?.SignatureStatus;
  const subject = candidate?.signerSubject ?? candidate?.SignerSubject;
  if (status !== "Valid" || typeof subject !== "string") return false;
  return subject.split(",").some((part) => part.trim().toLowerCase() === `cn=${CLI_SIGNER_COMMON_NAME}`);
}

const DISCOVERY_SHELLS = Object.freeze(["powershell.exe", "pwsh.exe"]);
const SHELL_TIMEOUT_MS = Math.floor(DISCOVERY_TIMEOUT_MS / DISCOVERY_SHELLS.length);

// This default is intentionally read-only and its raw command line stays local
// to this function. Tests inject listProcesses and never invoke PowerShell.
async function defaultListProcesses(signal) {
  const script = "$ErrorActionPreference='Stop'; Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'language_server*' -or $_.Name -eq 'agy.exe' } | ForEach-Object { $sig = $null; if ($_.Name -eq 'agy.exe' -and $_.ExecutablePath) { try { $sig = Get-AuthenticodeSignature $_.ExecutablePath } catch { $sig = $null } }; [pscustomobject]@{ ProcessId = $_.ProcessId; ExecutablePath = $_.ExecutablePath; CommandLine = $_.CommandLine; SignatureStatus = [string]$sig.Status; SignerSubject = [string]$sig.SignerCertificate.Subject } } | ConvertTo-Json -Compress";
  let lastError;
  let unsignedRows = null;
  for (const shell of DISCOVERY_SHELLS) {
    try {
      const { stdout } = await execFileAsync(shell, ["-NoProfile", "-NonInteractive", "-Command", script], {
        windowsHide: true, timeout: SHELL_TIMEOUT_MS, maxBuffer: 1024 * 1024, signal,
      });
      const parsed = JSON.parse(stdout || "[]");
      const rows = Array.isArray(parsed) ? parsed : [parsed];
      if (!needsSignatureRetry(rows)) return rows;
      unsignedRows = rows;
    } catch (error) {
      if (signal?.aborted) throw error;
      lastError = error;
    }
  }
  if (unsignedRows) return unsignedRows;
  throw lastError;
}

function needsSignatureRetry(rows) {
  return rows.some((row) => {
    const executablePath = row?.ExecutablePath;
    return typeof executablePath === "string" && executablePath && isAgyCliName(executablePath) && !row?.SignatureStatus;
  });
}

async function defaultPortsByPid(signal) {
  const { stdout } = await execFileAsync("netstat.exe", ["-ano", "-p", "tcp"], { windowsHide: true, timeout: 3_000, maxBuffer: 1024 * 1024, signal });
  const ports = [];
  for (const line of stdout.split(/\r?\n/)) {
    const match = line.match(/^\s*TCP\s+127\.0\.0\.1:(\d+)\s+\S+\s+LISTENING\s+(\d+)\s*$/i);
    if (match) ports.push({ pid: Number(match[2]), port: Number(match[1]) });
  }
  return (pid) => ports.filter((entry) => entry.pid === pid).map((entry) => entry.port);
}

async function verifyCandidate(candidate, inspectCandidate, signal, homeDir) {
  const pid = Number(candidate?.pid ?? candidate?.ProcessId);
  const executablePath = candidate?.executablePath ?? candidate?.ExecutablePath;
  const commandLine = candidate?.commandLine ?? candidate?.CommandLine;
  if (!Number.isInteger(pid) || pid <= 0 || typeof executablePath !== "string" || !executablePath) return { candidate: null, unsupported: false };
  if (isAgyCliName(executablePath)) {
    if (canonicalAgyCliPath(executablePath, homeDir) === null) return { candidate: null, unsupported: true };
    if (!isSignedByCliVendor(candidate)) return { candidate: null, unsupported: true };
    return { candidate: { pid, csrfToken: null }, unsupported: false };
  }
  if (flagValue(commandLine, "--app_data_dir") !== "antigravity-ide") return { candidate: null, unsupported: false };
  const canonicalExecutablePath = canonicalLanguageServerPath(executablePath, homeDir);
  if (canonicalExecutablePath === null) return { candidate: null, unsupported: true };
  const csrfToken = flagValue(commandLine, "--csrf_token");
  if (!csrfToken) return { candidate: null, unsupported: false };
  try {
    const proof = await inspectCandidate({ pid, executablePath: canonicalExecutablePath, candidate, signal });
    if (proof?.extensionName !== EXTENSION_IDENTITY.name || proof?.extensionPublisher !== EXTENSION_IDENTITY.publisher) return { candidate: null, unsupported: true };
    return { candidate: { pid, csrfToken }, unsupported: false };
  } catch {
    return { candidate: null, unsupported: true };
  }
}

function groupFromBuckets(displayName, expected) {
  const reasons = [expected.five?.reason, expected.seven?.reason].filter(Boolean);
  if (reasons.length === 0) return { displayName, five: expected.five?.window ?? null, seven: expected.seven?.window ?? null, status: "available" };
  const precedence = ["parse_error", "invalid_reset", "stale_reset", "window_unclassifiable"];
  const reason = precedence.find((candidate) => reasons.includes(candidate)) ?? "window_unclassifiable";
  const validCount = [expected.five?.window, expected.seven?.window].filter(Boolean).length;
  return {
    displayName,
    five: expected.five?.window ?? null,
    seven: expected.seven?.window ?? null,
    status: validCount ? "degraded" : "unavailable",
    reason,
  };
}

const PROBE_FAILURE_PRECEDENCE = ["auth_error", "rate_limited", "redirect_disallowed", "source_timeout", "parse_error", "source_error"];

function selectProbeFailure(results) {
  const reasons = results.map((result) => result?.reason).filter(Boolean);
  const reason = PROBE_FAILURE_PRECEDENCE.find((candidate) => reasons.includes(candidate)) ?? "source_error";
  return results.find((result) => result?.reason === reason) ?? failure(PROVIDER, SOURCE, reason, { target_origin: ORIGIN });
}

function normalizeAgyPayload(payload, nowMs) {
  const response = payload?.response ?? payload;
  const rawGroups = response?.quotaSummary?.groups ?? response?.groups ?? response?.quotaSummaryGroups;
  if (!Array.isArray(rawGroups)) return { reason: "parse_error" };
  const groups = {};
  for (const definition of AGY_GROUPS) {
    const rawGroup = rawGroups.find((group) => group?.displayName === definition.displayName);
    const normalized = {};
    if (!rawGroup) {
      groups[definition.id] = groupFromBuckets(definition.displayName, {
        five: { reason: "window_unclassifiable" },
        seven: { reason: "window_unclassifiable" },
      });
      continue;
    }
    for (const [key, bucketId] of Object.entries(definition.buckets)) {
      const sourceWindow = key === "five" ? "5h" : "weekly";
      const bucket = rawGroup?.buckets?.find((entry) => entry?.bucketId === bucketId);
      const window = bucket?.window === sourceWindow && normalizeWindow({ remainingFraction: bucket.remainingFraction, resetTime: bucket.resetTime, windowMinutes: key === "five" ? 300 : 10080, sourceWindow, nowMs });
      normalized[key] = window?.window ? window : { reason: window?.reason ?? "window_unclassifiable" };
    }
    groups[definition.id] = groupFromBuckets(definition.displayName, normalized);
  }
  return { groups };
}

export async function fetchAgyUsage(options = {}) {
  if ((options.platform ?? process.platform) !== "win32") return failure(PROVIDER, SOURCE, "unsupported_platform", { target_origin: ORIGIN });
  const listProcesses = options.listProcesses ?? defaultListProcesses;
  const homeDir = options.homeDir ?? userInfo().homedir;
  const inspectCandidate = options.inspectCandidate ?? (async ({ executablePath, signal }) => {
    const extensionRoot = dirname(dirname(executablePath));
    const readText = options.readText ?? ((filePath, encoding) => readFile(filePath, { encoding, signal }));
    const packageJson = JSON.parse(await readText(join(extensionRoot, "package.json"), "utf8"));
    return {
      extensionName: packageJson?.name,
      extensionPublisher: packageJson?.publisher,
    };
  });
  const outcome = await runWithDeadline({
    timeoutMs: options.timeoutMs,
    run: async (deadline) => {
      const discovery = await runWithDeadline({
        timeoutMs: Math.min(options.discoveryTimeoutMs ?? DISCOVERY_TIMEOUT_MS, deadline.remainingMs()),
        parentSignal: deadline.signal,
        run: async (discoveryDeadline) => {
          let candidates;
          try {
            candidates = await listProcesses(discoveryDeadline.signal);
          } catch {
            return failure(PROVIDER, SOURCE, "local_rpc_unavailable", { target_origin: ORIGIN });
          }
          if (discoveryDeadline.signal.aborted) return failure(PROVIDER, SOURCE, "source_timeout", { target_origin: ORIGIN });
          const verified = [];
          let unsupported = false;
          for (const candidate of Array.isArray(candidates) ? candidates : []) {
            const candidateResult = await verifyCandidate(candidate, inspectCandidate, discoveryDeadline.signal, homeDir);
            if (discoveryDeadline.signal.aborted) return failure(PROVIDER, SOURCE, "source_timeout", { target_origin: ORIGIN });
            if (candidateResult.candidate) verified.push(candidateResult.candidate);
            unsupported ||= candidateResult.unsupported;
          }
          if (verified.length === 0) return failure(PROVIDER, SOURCE, unsupported ? "unsupported_version" : "local_rpc_unavailable", { target_origin: ORIGIN });
          let resolver;
          try {
            resolver = options.portsByPid ?? await defaultPortsByPid(discoveryDeadline.signal);
          } catch {
            return failure(PROVIDER, SOURCE, "local_rpc_unavailable", { target_origin: ORIGIN });
          }
          const targets = [];
          for (const candidate of verified) {
            let ports;
            try {
              ports = await resolver(candidate.pid);
            } catch {
              return failure(PROVIDER, SOURCE, "local_rpc_unavailable", { target_origin: ORIGIN });
            }
            if (discoveryDeadline.signal.aborted) return failure(PROVIDER, SOURCE, "source_timeout", { target_origin: ORIGIN });
            if (Array.isArray(ports)) {
              for (const port of ports) {
                if (Number.isInteger(port) && port >= 1 && port <= 65535) targets.push({ candidate, port });
              }
            }
          }
          if (targets.length === 0) {
            return failure(PROVIDER, SOURCE, "local_rpc_unavailable", { target_origin: ORIGIN });
          }
          return { targets };
        },
      });
      if (discovery.timedOut) return failure(PROVIDER, SOURCE, "source_timeout", { target_origin: ORIGIN });
      if (discovery.error) return failure(PROVIDER, SOURCE, "local_rpc_unavailable", { target_origin: ORIGIN });
      if (!discovery.value.ok && discovery.value.ok !== undefined) return discovery.value;
      const probes = discovery.value.targets.map(({ candidate, port }) => {
        const cancel = new AbortController();
        const probeDeadline = createDeadline(deadline.remainingMs(), AbortSignal.any([deadline.signal, cancel.signal]));
        const promise = requestJson({
          request: options.request,
          url: `http://127.0.0.1:${port}${RPC_PATH}`,
          provider: PROVIDER,
          source: SOURCE,
          targetOrigin: ORIGIN,
          timeoutMs: options.timeoutMs,
          deadline: probeDeadline,
          init: { method: "POST", headers: { "content-type": "application/json", ...(candidate.csrfToken ? { "x-codeium-csrf-token": candidate.csrfToken } : {}) }, body: "{}" },
        }).finally(() => probeDeadline.close());
        return { cancel, promise };
      });
      const result = await new Promise((resolve) => {
        let remaining = probes.length;
        const failedResults = [];
        for (const probe of probes) {
          void probe.promise.then((probeResult) => {
            if (probeResult.ok) {
              for (const other of probes) if (other !== probe) other.cancel.abort();
              resolve(probeResult);
              return;
            }
            failedResults.push(probeResult);
            if (--remaining === 0) resolve(selectProbeFailure(failedResults));
          }).catch(() => {
            failedResults.push(failure(PROVIDER, SOURCE, "source_error", { target_origin: ORIGIN }));
            if (--remaining === 0) resolve(selectProbeFailure(failedResults));
          });
        }
      });
      if (!result.ok) return result;
      const normalized = normalizeAgyPayload(result.value, options.nowMs ?? Date.now());
      return normalized.groups ? { ok: true, provider: PROVIDER, source: SOURCE, groups: normalized.groups } : failure(PROVIDER, SOURCE, normalized.reason, { target_origin: ORIGIN });
    },
  });
  if (outcome.timedOut) return failure(PROVIDER, SOURCE, "source_timeout", { target_origin: ORIGIN });
  if (outcome.error) return failure(PROVIDER, SOURCE, "local_rpc_unavailable", { target_origin: ORIGIN });
  return outcome.value;
}
