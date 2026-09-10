#!/usr/bin/env node
/**
 * usage.mjs — 벤더 사용량을 리셋 창 대비로 판정한다.
 *
 * madi의 usage-guard(자체 제작, R72a에서 외부 도구 의존을 걷어냄)를 호출해
 * raw JSON을 받고, **잔여율과 시간 잔여율을 대비**해 실제 여유를 판정한다.
 *
 * 왜 대비가 필요한가: 잔여 퍼센트만 보면 틀린다. 2026-08-12 실측에서
 * Codex 51% / Claude 25%였는데, 리셋 대비로는 Codex가 더 빠듯했다 —
 * Codex는 시간이 82% 남았는데 사용량이 51%뿐이라 이미 페이스를 초과했고,
 * Claude는 25%만 남았지만 시간도 21%만 남아 페이스로는 오히려 여유였다.
 *
 *   left > timeLeftRatio → 페이스보다 덜 씀 = 여유
 *   left < timeLeftRatio → 페이스보다 많이 씀 = 빠듯
 *
 * 사용:
 *   node tools/usage.mjs            사람이 읽는 표
 *   node tools/usage.mjs --json     기계 판독(원본 + 판정)
 *
 * madi 경로는 환경변수 `MADI_ROOT`로 덮을 수 있다(기본 D:/Code/madi).
 * Antigravity는 **IDE가 실행 중일 때만** 계측된다(agy-local-rpc).
 */

import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";

const MADI_ROOT = process.env.MADI_ROOT || "D:/Code/madi";
const SNAPSHOT = path.join(MADI_ROOT, "tools", "usage-guard", "snapshot-all.mjs");
const AS_JSON = process.argv.includes("--json");

/** 여유 판정. margin이 양수면 페이스보다 덜 쓴 것. */
function verdict(left, timeLeftRatio) {
  if (left == null || timeLeftRatio == null) return null;
  const margin = left - timeLeftRatio;
  let label;
  if (margin >= 0.15) label = "여유";
  else if (margin >= 0.0) label = "정상";
  else if (margin >= -0.15) label = "빠듯";
  else label = "위험";
  return { margin, label };
}

function pct(x) {
  return x == null ? "  —  " : `${(x * 100).toFixed(0).padStart(3)}%`;
}

function rows(snap) {
  const out = [];
  const push = (name, w) => {
    if (!w) return;
    out.push({
      vendor: name,
      window: w.sourceWindow ?? "?",
      left: w.left ?? w.remainingFraction ?? null,
      timeLeft: w.timeLeftRatio ?? null,
      resetAt: w.resetAt ?? null,
      ...(verdict(w.left ?? w.remainingFraction, w.timeLeftRatio) ?? {}),
    });
  };
  const p = snap.providers ?? {};
  for (const [name, v] of Object.entries(p)) {
    if (name === "antigravity") {
      for (const [g, gv] of Object.entries(v.groups ?? {})) {
        // 두 source의 모양이 다르다 — `checkpoint.mjs`는 `{status, windows:{five,seven}}`,
        // `snapshot-all.mjs`는 `{five, seven, status}`로 창을 그룹에 직접 둔다.
        // status 어휘도 갈린다("pass" vs "available"). 라벨과 래핑을 믿지 말고
        // **창이 실제로 왔는지**로 판정한다.
        const w = gv.windows ?? gv;
        if (!w.five && !w.seven) {
          out.push({ vendor: `agy:${g}`, window: "-", unavailable: gv.reason ?? gv.status ?? "no_windows" });
          continue;
        }
        push(`agy:${g}`, w.five);
        push(`agy:${g}`, w.seven);
      }
      continue;
    }
    // agy만이 아니라 claude·codex도 `windows`로 감싼다 — 래핑 처리를 agy 분기 안에만
    // 두는 바람에 두 벤더가 표에서 통째로 빠져 있었다(2026-08-12).
    const w = v.windows ?? v;
    push(name, w.five);
    push(name, w.seven);
  }
  return out;
}

let raw;
try {
  if (!existsSync(SNAPSHOT)) {
    console.error(`usage-guard를 찾을 수 없다: ${SNAPSHOT}`);
    console.error("MADI_ROOT 환경변수로 madi 저장소 경로를 지정하라.");
    process.exit(2);
  }
  raw = JSON.parse(execFileSync("node", [SNAPSHOT], { encoding: "utf8", timeout: 120_000 }));
} catch (err) {
  console.error(`사용량 조회 실패: ${err.message}`);
  process.exit(1);
}

// Antigravity 그룹이 하나라도 계측 불가면 그 사실을 눈에 띄게 남긴다 —
// 모르는 채로 쏘는 것이 잔여를 아는 것보다 나쁘다.
const agyDown = Object.values(raw.providers?.antigravity?.groups ?? {}).some(
  (g) => !(g.windows ?? g)?.five && !(g.windows ?? g)?.seven,
);

const table = rows(raw);

if (AS_JSON) {
  console.log(JSON.stringify({ ts: raw.ts, rows: table, raw }, null, 2));
} else {
  console.log(`\n사용량 스냅샷 — ${raw.ts}\n`);
  console.log("벤더            창       잔여   시간잔여  판정    리셋");
  console.log("─".repeat(64));
  for (const r of table) {
    if (r.unavailable) {
      console.log(`${r.vendor.padEnd(15)} ${"-".padEnd(8)} 계측 불가 (${r.unavailable})`);
      continue;
    }
    const reset = r.resetAt ? r.resetAt.replace("T", " ").slice(0, 16) : "—";
    console.log(
      `${r.vendor.padEnd(15)} ${String(r.window).padEnd(8)} ${pct(r.left)}  ${pct(r.timeLeft)}   ${(r.label ?? "").padEnd(5)} ${reset}`,
    );
  }
  console.log(
    "\n판정 = 잔여 − 시간잔여. 양수면 페이스보다 덜 쓴 것이다.\n" +
      "  여유 ≥ +15%p · 정상 0~15%p · 빠듯 0~−15%p · 위험 < −15%p",
  );
  if (agyDown) {
    console.log(
      "\n⚠️ Antigravity 계측 불가 — IDE가 실행 중이어야 읽힌다(agy-local-rpc).\n" +
        "   이 상태로 AG를 호출하면 잔여를 모르는 채 소모한다.",
    );
  }
}
