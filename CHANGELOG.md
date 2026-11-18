# Changelog

All notable public-release changes are documented here.

## [0.1.5] - 2026-11-18

Fix-only release: the dispatcher path fix below. Without `SECOND_OPINION_DISPATCH`,
0.1.4 produced no notes. Note generation is unchanged, so note quality is the same
as in 0.1.4.

### Fixed

- The second-opinion dispatcher no longer needs `SECOND_OPINION_DISPATCH`. NoteFactory
  looks for it in this order: that variable, the highest installed version under
  `$CODEX_HOME` (or `~/.codex`), the highest under `~/.claude`, then the Claude Code
  marketplace copy. A machine-specific default path that no longer existed made every
  API call fail without the variable. The writer route and the premium arbiter now use
  the same lookup.

## [0.1.4] - 2026-11-16

### Changed

- Transcript (video) notes now receive the video description and metadata (title,
  channel, publish date) in their synthesis source. The transcript writer prompt
  forbids claims, applications, or dates that are not in the source, and asks the
  writer to keep qualifiers, required thresholds, speaker attribution, and the unit
  of each statement.
- Transcripts longer than 12,000 characters are split into chunks of about 10,000
  characters (a transcript without line breaks is split at sentence ends). Each
  chunk runs through the note harness and is told it is chunk n of N. The chunk
  notes are then regrouped into one note by topic: a model call, asked three times,
  proposes topic groups, an opening summary, and which sections belong to the
  operations/announcements section (only sections all three answers agree on move
  there). Code copies the chunk body sentences verbatim and orders groups by the
  earliest source timestamp. If regrouping fails, the chunk notes are merged in
  source order.
- On the transcript path the same-model repair pass is off: critic findings and a
  number-qualifier check are kept as run diagnostics and do not rewrite the note.
  `text_post` still uses up to two repair passes.
- Transcript-only timestamp and structure checks read the original transcript
  instead of the combined source.
- The shared timestamp gate pads one-digit minutes (`[t=1:6:07]` becomes
  `[t=01:06:07]`), keeps the time part of comma-joined values, and drops truncated
  values.
- Chunked notes: the header `passes_run` lists each chunk's passes plus `regroup`
  when regrouping succeeds, token totals include regroup calls (also calls made
  before a regroup failure), and `model_reported` is a single value or a readable
  `mixed(...)` label.
- After merging, cleanup removes only sentences that literally contain the chunk
  notice NoteFactory inserted (`[구간 안내]` or its fixed sentence) and link lines
  whose only URL is an ellipsis placeholder such as `watch?v=...`. Timestamps in
  section titles and repeated link lines are tidied, and a link line whose only
  description says none was given keeps just the URL.

### Known limitations

These apply to transcript notes on the default free writer route and were accepted
for this release:

- Qualifiers can still be dropped (one or two in a long lecture note); the
  diagnostic check catches only some of them.
- Timestamps point to the vicinity of a statement; several statements can share one.
- In step-by-step lectures, the chunk plan can merge several steps into one topic,
  which shortens per-step detail. Results vary between runs.
- Content on a chunk boundary can appear twice.
- Repetitive "the speaker says" phrasing and a few operational remarks inside content
  sections can remain.
- Sentences that restate the chunk notice without its literal marker, and real-format
  URLs that are not in the source, are not removed.

## [0.1.3] - 2026-10-24

### Added

- Sipher v0.1.8 label values are recognized in the collection-status block:
  `transcript_label` values `no_audio`, `not_requested`, `no_speech`, and
  `skipped_ambient_audio`, and `ocr_label` value `not_requested` now print their
  meaning without being counted as deficits. The `not_requested`, `no_speech`,
  and ambient-audio lines state that the corresponding content is not in the note.
- `.env.example` includes the optional `NVIDIA_NIM_API_KEY` entry.

### Fixed

- Those five label values previously fell through to unreadable-status messages
  (`TRANSCRIPT_UNREADABLE` or `OCR_UNREADABLE`).
- Corrected the v0.1.2 README guidance: all API routes, including default Gemini
  and OpenRouter, require Node.js and `SECOND_OPINION_DISPATCH`. Quick start now
  begins with step 0 to prepare Node.js and the dispatcher.

### Changed

- A Sipher result with `transcript_label=no_speech` and empty `transcript` and
  `body_text` now ends without a note or any model call, with exit code 3
  ("no source to synthesize"; rerunning the same input gives the same result),
  instead of an error. `note_batch` records these items as a separate
  `no_synthesis_source` outcome: not a failure, never reassigned or promoted,
  counted in `summary.json` and the completion log, and not used to set the batch
  exit code.
- NVIDIA NIM calls now also go through the second-opinion dispatcher;
  `SECOND_OPINION_DISPATCH` is required for NIM routes. NoteFactory's own NIM
  rate limiter (30 RPM, two concurrent calls) and retry loop have been removed;
  rate limiting and retries are handled by the dispatcher.

## [0.1.2] - 2026-10-08

### Added

- `--no-critic-fallback` pins the critic for controlled benchmarks. With the flag, the
  critic client holds only the requested critic model. If that critic fails, NoteFactory
  does not call another free critic, the paid critic tail, or the premium arbiter. The
  long-material retry keeps the same setting. Without the flag, critic fallback and
  default generation behave as in v0.1.1.

## [0.1.1] - 2026-09-10

### Added

- Public release surface: setup scripts, safe archive builder, version file, example
  environment configuration, English and Korean READMEs, and MIT license.
- Gemini Lite-first writer/critic configuration documented as the default free path.
- A harness engineering playbook and benchmark-case catalog for development users.

### Changed

- Source-preservation evaluation now separates deterministic URL/number/literal
  contracts from semantic advisories, so natural paraphrases are not rejected solely
  for keyword mismatch.
- The production number gate preserves sequence and order without requiring titles to
  repeat the source's exact wording.
- Critic structured-output parsing retries bounded transient format failures and
  benchmark subprocesses preserve UTF-8 diagnostics without opening a Windows console.

### Fixed

- URL comparison accepts optional schemes, Markdown backticks, and a later repeated
  URL occurrence that supplies its explanation.
- A failed Gemini anchor note is classified as a harness investigation signal instead
  of silently becoming a model-quality verdict.
- Public documentation now distinguishes the free portable starter configuration from
  the maintainer-only local CLI-subscription fallback, and keeps the non-portable
  `--vault` integration out of public quick-start guidance.
- Public archives exclude benchmark/probe tooling and test-only vendor JavaScript.
- A fresh public checkout can run CLI help before `.env.local` exists; generation
  still fails later with the normal inactive-provider diagnostic until configured.

## [0.1.0]

- Initial private development archive baseline.
