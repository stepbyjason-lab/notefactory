# Changelog

All notable public-release changes are documented here.

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
