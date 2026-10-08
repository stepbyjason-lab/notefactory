<!-- Language: Korean · [English](README.md) -->

# NoteFactory

> **현재 공개 버전: v0.1.2**

Sipher가 정규화한 JSON을 한국어 지식노트 Markdown으로 바꾸는 로컬 파이프라인입니다.

```text
URL / 파일 ── Sipher ──> 정규화 JSON ── NoteFactory ──> 지식노트
```

Sipher는 원문을 수집하고 수집 상태를 정직하게 라벨링합니다. NoteFactory는 그 자료를
학습 가능한 노트로 재구성하고, URL·번호·복사용 프롬프트처럼 빠지면 쓸 수 없는 정보를
보전하며, 최종 gate 상태를 노트 헤더에 기록합니다.

기본 경로는 무료 모델입니다.

## 할 수 있는 일

- Sipher 8-key 정규화 JSON을 파일 또는 stdin으로 읽습니다.
- writer → 결정적 gate → critic → 필요 시 repair 순서로 노트를 만듭니다.
- URL과 설명, 원문 번호 순서, 긴 프롬프트·템플릿·코드 같은 복사용 자료를 보전합니다.
- 노트 헤더에 source, 모델 route, 실행 패스, 토큰 보고, `verified` 상태를 남깁니다.
- 무료 writer 후보 목록을 순서대로 시도하고 실제 fallback을 정직하게 기록합니다.
- `--vault`를 명시하지 않으면 SecondBrain/Obsidian 볼트에 쓰지 않습니다.

## 하지 않는 일

- URL을 직접 수집하지 않습니다. 먼저 [Sipher](https://github.com/stepbyjason-lab/sipher)를
  설치하거나 이미 정규화된 JSON을 넣어야 합니다.
- `verified: True`가 모든 의미 해석과 산문 품질이 완벽하다는 보증은 아닙니다. 이는 설정된
  production pipeline과 결정적 원문 보전 gate가 통과했다는 뜻입니다.
- 공개 starter 설정에는 무료 Gemini와 선택적 OpenRouter free route만 들어 있습니다. 사용자가
  과금되는 provider를 직접 후보 목록에 추가하면 그것은 사용자의 명시 설정이며 비용이 발생할 수
  있습니다.
- `--allow-paid-fallback`은 portable API-key fallback이 아니라 개발자 로컬 CLI 구독 연동입니다.
  해당 provider 도구와 Node.js가 필요하며 공개 기본 setup에는 필요하지 않습니다.
- 여러 계정이나 여러 키로 무료 한도를 우회하지 않습니다.

## 빠른 시작

### 1. 설치

```bash
git clone https://github.com/stepbyjason-lab/notefactory.git
cd notefactory

# Windows PowerShell
scripts/setup.ps1

# macOS / Linux
scripts/setup.sh
```

설치 스크립트는 `.venv`를 만들고, 필요한 Python 의존성을 설치하며, `.env.local`이 없으면
`.env.example`을 복사합니다.

### 2. 무료 기본 writer 설정

`.env.local`에 Gemini 키를 입력합니다.

```dotenv
GEMINI_API_KEY=your_key_here
```

기본 무료 순서는 다음과 같습니다.

```text
Writer
1. gemini/gemini-3.5-flash-lite
2. gemini/gemini-3.1-flash-lite
3. openrouter/google/gemma-4-31b-it:free   # 설정했을 때만

Critic
1. gemini/gemma-4-31b-it
2. gemini/gemini-3.1-flash-lite
3. gemini/gemini-3.5-flash-lite
```

`WRITER_CANDIDATES`, `CRITIC_CANDIDATES`는 콤마 구분 `provider:model` 목록입니다.
`.env.local`의 순서만 바꾸면 코드 수정 없이 우선순위를 바꿀 수 있습니다.

### 3. Sipher로 입력 수집

```bash
# Sipher 체크아웃에서 실행합니다.
python -m core fetch "https://www.threads.net/@someone/post/POST_ID" --json --out source.json
```

OCR, 전사, Threads 연속글, 로그인 플랫폼은 Sipher의 최신 공개 문서를 따르세요.

### 4. 노트 생성

```bash
# Windows
.venv\Scripts\python.exe note_pipe.py source.json --out notes_out

# macOS / Linux
.venv/bin/python note_pipe.py source.json --out notes_out
```

저장 경로가 출력됩니다. 노트 헤더의 `verified: True`는 최종 production gate가 통과했다는
뜻입니다. `verified: False`여도 노트와 남은 finding을 보존하므로 실패 원인을 직접 확인할 수
있습니다.

## 자주 쓰는 명령

```bash
# 특정 writer만 고정해 실험할 때
python note_pipe.py source.json --out notes_out --provider gemini --model gemini-3.5-flash-lite --no-writer-fallback

# critic까지 고정할 때. 요청한 critic이 실패해도 다른 critic 모델을 부르지 않습니다.
python note_pipe.py source.json --out notes_out --provider gemini --model gemini-3.5-flash-lite --no-writer-fallback --critic-provider gemini --critic-model gemma-4-31b-it --no-critic-fallback

# 빠른 경량 경로
python note_pipe.py source.json --out notes_out --profile light

```

전체 옵션은 `python note_pipe.py --help`를 확인하세요.

`--vault`는 v0.1.2에서 maintainer의 로컬 Windows 경로에 연결된 내부 개발 통합입니다.
공개 사용자는 항상 명시적인 `--out` 디렉터리를 사용하세요.

## 원문 보전 원칙

노트는 전사지가 아니지만, 아래 정보는 요약 과정에서 사라지면 안 됩니다.

- 원문 URL과 그 URL의 역할 설명
- 원저자가 붙인 번호와 순서
- 복사해 써야 하는 프롬프트·템플릿·코드

NoteFactory는 실제 누락과 자연스러운 표현 차이를 분리합니다. 다른 말로 잘 설명한 것을
단어 불일치만으로 탈락시키지 않지만, URL·번호·필수 원문 자료가 실제로 빠지면 명시적으로
실패시킵니다.

## 한계와 안전

- 무료 provider의 busy, timeout, quota는 언제든 달라질 수 있습니다.
- `verified`는 원문 보전 gate 상태이지 새로운 모델의 의미 품질을 자동 보증하는 점수가 아닙니다.
- 공개 콘텐츠 또는 provider 전송 위험을 수용할 수 있는 자료만 넣으세요.
- 키는 `.env.local`에만 두고 커밋·터미널·결과 파일에 넣지 마세요. starter 설정 밖의
  provider를 직접 추가할 때는 해당 provider의 과금·약관도 별도로 확인하세요.
- 공개 배포판에는 내부 handoff, benchmark 입력·결과, 로컬 agent 설정, 테스트 fixture를 넣지
  않습니다.

## 라이선스

MIT. [LICENSE](LICENSE)를 참고하세요.
