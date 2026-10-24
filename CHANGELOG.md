# Changelog

All notable public-release changes are documented here.

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
