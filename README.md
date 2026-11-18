<!-- Language: English · [한국어](README.ko.md) -->

# NoteFactory

> **Current public release: v0.1.5**

Turn Sipher's normalized JSON into a usable Korean knowledge note.

NoteFactory is the writing half of a two-stage local workflow:

```text
URL or file ── Sipher ──> normalized JSON ── NoteFactory ──> Markdown note
```

Sipher collects and labels what it could actually recover. NoteFactory turns that
material into a structured note, preserves material that must remain copyable, and
records whether its deterministic source-preservation gates passed.

It is designed for public content and free-tier model routes by default.

## What it does

- Reads Sipher's normalized 8-key JSON from a file or standard input.
- Generates a Korean knowledge note through a writer → gate → critic → conditional
  repair pipeline.
- Writes long video transcripts (over 12,000 characters) chunk by chunk and regroups
  the result by topic. On the transcript path, critic findings are kept as diagnostics
  instead of triggering a same-model repair.
- Preserves source URLs with nearby explanations, numbered source sequences, and
  long prompts/templates/code that should remain copyable.
- Stores provenance in the note header: source, requested/observed model route,
  passes run, token report, and `verified` state.
- Uses a free writer priority chain by default and records fallback honestly.
- Never writes to an Obsidian/SecondBrain vault unless `--vault` is explicitly used.

## What it does not do

- It does not scrape URLs itself. Install and run [Sipher](https://github.com/stepbyjason-lab/sipher)
  first, or provide an already normalized JSON file.
- It does not turn a hard-contract pass into a guarantee of perfect prose or factual
  interpretation. `verified: True` means the configured deterministic pipeline
  completed and its source-preservation gates passed.
- The public starter configuration contains only free Gemini and optional OpenRouter
  free routes. If you manually add a provider whose account bills, that is your
  explicit configuration and can incur charges.
- `--allow-paid-fallback` is a developer-only local CLI-subscription integration,
  not a portable API-key fallback. It requires the matching local provider tooling
  in addition to Node.js, which is also required for the public default API routes.
- It does not bypass provider quotas or automate multiple-account rotation.

## Quick start

### 0. Prepare Node.js and the dispatcher

The dispatcher is part of the public [second-opinion plugin](https://github.com/stepbyjason-lab/second-opinion) (MIT).
Install it in Claude Code with `/plugin marketplace add stepbyjason-lab/second-opinion`
and `/plugin install second-opinion@second-opinion`, or clone it with
`git clone https://github.com/stepbyjason-lab/second-opinion`.
In the repository, the dispatcher is `plugins/second-opinion/scripts/dispatch.mjs`;
after a plugin install, it is `~/.claude/plugins/cache/second-opinion/second-opinion/<version>/scripts/dispatch.mjs`.

Install Node.js. NoteFactory finds the dispatcher on its own, in this order:

1. the process environment variable `SECOND_OPINION_DISPATCH`, if set;
2. the highest installed version under `$CODEX_HOME/plugins/cache/second-opinion/second-opinion/`
   (`~/.codex/...` when `CODEX_HOME` is unset);
3. the highest installed version under `~/.claude/plugins/cache/second-opinion/second-opinion/`;
4. the Claude Code marketplace copy of the plugin.

With a plugin install you need nothing else. If you use a cloned repository instead,
point `SECOND_OPINION_DISPATCH` at it in the shell used to run NoteFactory
(a process environment variable, not a `.env.local` value):

```powershell
# Windows PowerShell
$env:SECOND_OPINION_DISPATCH = "/path/to/second-opinion/scripts/dispatch.mjs"
```

```bash
# macOS / Linux
export SECOND_OPINION_DISPATCH="/path/to/second-opinion/scripts/dispatch.mjs"
```

If none of these exist, every API call stops with "second-opinion 디스패처를 찾을 수 없다"
and the location it searched.

### 1. Clone and install

```bash
git clone https://github.com/stepbyjason-lab/notefactory.git
cd notefactory

# Windows PowerShell
scripts/setup.ps1

# macOS / Linux
scripts/setup.sh
```

The setup script creates `.venv`, installs the small runtime dependency set, and
copies `.env.example` to `.env.local` when no local configuration exists.

### 2. Configure a free writer route

Add a Gemini API key to `.env.local`:

```dotenv
GEMINI_API_KEY=your_key_here
```

The default free configuration is:

```text
Writer
1. gemini/gemini-3.5-flash-lite
2. gemini/gemini-3.1-flash-lite
3. openrouter/google/gemma-4-31b-it:free   # only when configured

Critic
1. gemini/gemma-4-31b-it
2. gemini/gemini-3.1-flash-lite
3. gemini/gemini-3.5-flash-lite
```

You can reorder `WRITER_CANDIDATES` and `CRITIC_CANDIDATES` in `.env.local`.
They are comma-separated `provider:model` lists, so no source-code edit is needed
to change priority.

All API routes, including the default Gemini route and OpenRouter, use the
second-opinion dispatcher and require Node.js and a dispatcher that step 0 can
find. NVIDIA NIM calls also use this
path; provide `NVIDIA_NIM_API_KEY` to use a NIM route.

### 3. Collect a source with Sipher

```bash
# Run this in the Sipher checkout.
python -m core fetch "https://www.threads.net/@someone/post/POST_ID" --json --out source.json
```

Use Sipher's current public documentation for platform-specific options such as OCR,
transcription, Threads continuation collection, or authenticated sources.

NoteFactory handles Sipher v0.1.8 output, including `transcript_label` values `no_audio`, `not_requested`, `no_speech`,
and `skipped_ambient_audio`, and `ocr_label=not_requested`.

### 4. Generate a note

```bash
# Windows
.venv\Scripts\python.exe note_pipe.py source.json --out notes_out

# macOS / Linux
.venv/bin/python note_pipe.py source.json --out notes_out
```

`note_pipe.py` prints the saved path. The note header records `verified: True` only
when the final production gates pass. A note may be saved with `verified: False` so
you can inspect the output and the remaining findings instead of losing evidence.

If a Sipher result has `transcript_label=no_speech` and both `transcript` and `body_text` are empty or whitespace-only strings, `note_pipe.py` exits with code 3
(no source to synthesize), without a note or any model call. This is not a failure; rerunning the same input gives the same result.
Other processing failures still exit with code 1. `note_batch` counts these no-source items separately as `no_synthesis_source`.

## Common commands

```bash
# Explicit writer model, without writer fallback. Useful for a controlled test.
python note_pipe.py source.json --out notes_out --provider gemini --model gemini-3.5-flash-lite --no-writer-fallback

# Also pin the critic. If the requested critic fails, no other critic model is called.
python note_pipe.py source.json --out notes_out --provider gemini --model gemini-3.5-flash-lite --no-writer-fallback --critic-provider gemini --critic-model gemma-4-31b-it --no-critic-fallback

# Lightweight pipeline: synthesis + deterministic gates + conditional repair.
python note_pipe.py source.json --out notes_out --profile light

```

Run `python note_pipe.py --help` for the complete CLI contract.

`--vault` is an internal development integration: its destination is
currently tied to the maintainer's local Windows layout and is not portable. Public
users should always choose an explicit `--out` directory.

## Source preservation rules

Some source material must remain available to the reader instead of being summarized
away. NoteFactory treats these as explicit preservation contracts where a benchmark
or a configured fixture requires them:

- URLs remain as URLs and need nearby explanations.
- Numbered source sequences keep their number and order.
- Long prompts, templates, and code can be preserved as copyable material while the
  surrounding explanation is rewritten as a note.

The project separates this deterministic preservation layer from semantic evaluation.
A natural paraphrase should not fail merely because it uses different words, while an
actually missing URL, number, or required source artifact must still fail loudly.

## Reliability and limits

Free provider availability changes. A busy response, timeout, rate limit, malformed
structured critic response, and a bad final note are different failures and are
recorded separately. NoteFactory retries bounded transient failures and can use the
next configured free route; it does not claim that a provider quota is unlimited.

The current benchmark harness records hard preservation contracts, semantic advisory
signals, execution time, routes, and final artifacts. Its automatic qualification is
a screening signal for known routes, not a substitute for human review when promoting
an entirely new model to a production recommendation.

Transcript notes have known limitations: a qualifier can be dropped,
timestamps point to the vicinity of a statement, step-by-step lectures can lose
per-step detail, chunk-boundary content can repeat, and restated chunk notices or
real-format URLs not in the source are not removed. See `CHANGELOG.md`.

## Privacy and security

- Send only public content or material whose provider-data risk you accept.
- Keep keys in `.env.local`; it is ignored by Git.
- Do not paste keys into terminal commands, issue reports, or benchmark artifacts.
- Review provider terms and quotas before sustained use, especially if you add a
  provider outside the free public starter configuration.
- The public release intentionally excludes development handoffs, benchmark inputs,
  local agent configuration, test fixtures, and cached outputs.

## Development and benchmarking

The development repository contains additional internal evidence and benchmark tools.
The public archive intentionally omits those materials. If you are evaluating a new
model, keep the input JSON, prompts, fixture, final note, route provenance, elapsed
time, and failure records together. Do not compare scores produced by different
harness inputs or configurations as though they were the same experiment.

## License

MIT. See [LICENSE](LICENSE).
