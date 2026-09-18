# Fork changes

이 저장소는 [langchain-ai/open-swe](https://github.com/langchain-ai/open-swe)의 포크(asomegs/open-swe)입니다.
upstream과 달라진 점을 이 문서에 기록합니다. 포크 전용 변경을 추가하면 아래 "변경 목록"에 항목을 더하고,
upstream이 같은 기능을 제공하게 되면 항목을 지우거나 "제거됨"으로 표시합니다.

## 운영 규칙

- `main`에서 작업하고, 주기적으로 `git fetch upstream && git merge upstream/main`으로 동기화합니다. rebase는 쓰지 않습니다.
- 커스터마이징 하나는 커밋 하나로 squash하고, 제목에 `(fork)` 스코프를 붙입니다. 예: `feat(fork): ...`
- 내 변경만 보기: `git log --no-merges --oneline upstream/main..main`
- 동기화할 때마다 `fork-sync-YYYY-MM-DD` 태그를 남기고 아래 표에 한 줄 추가합니다.
- 가능하면 범용적인 변경은 upstream에 PR로 올려 포크에서 유지할 패치를 줄입니다.

## 동기화 이력

| 날짜 | upstream 커밋 | 비고 |
|---|---|---|
| 2026-09-19 | `b61d8c244` | 포크 시작 지점 |

## 변경 목록

### 1. Codex CLI의 ChatGPT 로그인으로 OpenAI 모델 실행 (2026-09-19)

- **목적**: `OPENAI_API_KEY` 없이, 이 머신의 Codex CLI가 로그인해 둔 `~/.codex/auth.json`으로 `openai:` 모델을 씁니다. upstream에는 데스크톱 앱 전용 ChatGPT OAuth 브로커 경로만 있었습니다.
- **진입점**
  - `agent/utils/openai_oauth.py`: 파일 기반 토큰 프로바이더 `_CodexCliTokenProvider`, 두 경로를 아우르는 `openai_oauth_available` / `build_openai_oauth_model`, 구조화 출력 우회용 `_OpenSWEChatOpenAICodex`
  - `agent/utils/model.py`: `make_model`의 OpenAI OAuth 분기와 localhost 기동 검증이 위 함수를 사용
  - `agent/config.py`: 환경변수 `OPEN_SWE_CODEX_AUTH_FILE`
  - `agent/dashboard/workspace_settings.py`, `agent/sandboxes/providers/local.py`: 각 한 줄
- **설정** (`docs/INSTALLATION.md` 4절, `docs/DEVELOPMENT.md` 5절)
  - `OPEN_SWE_CODEX_AUTH_FILE="~/.codex/auth.json"`, `OPENAI_API_KEY`는 비움, 게이트웨이 꺼짐
  - `LLM_MODEL_ID`와 `LLM_FALLBACK_MODEL_ID`를 `openai:` 모델로 지정. Admin → Global defaults의 에이전트·라우팅·제목 모델도 OpenAI로 맞춰야 기본 경로가 Codex 로그인을 탑니다.
- **설계상 선택**
  - 파일은 읽기 전용이며 mtime/크기가 바뀌면 재파싱합니다. 토큰 갱신은 Codex CLI에 맡깁니다. langchain이 `~/.codex/auth.json`에 대한 refresh 회전이 CLI 세션을 깨뜨릴 수 있다고 경고하고, 액세스 토큰 수명이 10일이라 충분합니다. 만료되면 `codex login` 안내가 담긴 오류로 실패합니다.
  - 데스크톱 브로커와 파일이 둘 다 설정되면 브로커가 우선합니다.
  - Codex 백엔드는 `json_schema` 구조화 출력의 JSON은 돌려주지만 스트리밍 완료 응답에 `text.format`을 되돌려주지 않아 langchain이 `parsed`를 채우지 않습니다. 그래서 `_OpenSWEChatOpenAICodex.with_structured_output`이 `json_schema` 요청을 `function_calling`으로 바꿉니다. 스레드 제목, 모델 라우팅 분류기, 리뷰 diff 그룹화, 데스크톱 브랜치명이 이 경로를 씁니다. 데스크톱 브로커 경로도 같은 클래스를 씁니다.
- **충돌 위험**: `agent/utils/model.py`의 `make_model` 분기와 `agent/config.py`가 upstream에서 자주 바뀝니다. `server.py`는 건드리지 않았습니다.
- **upstream 가능성**: 높습니다. 데스크톱 OAuth 경로를 서버 모드로 확장한 것이라 그대로 PR 후보입니다. 구조화 출력 우회는 langchain-openai의 `_ChatOpenAICodex` 이슈로 올릴 대상이고, 거기서 고쳐지면 서브클래스를 제거합니다.
- **테스트**: `tests/models/test_openai_oauth.py`

### 2. LiteLLM 등 OpenAI 호환 게이트웨이의 모델 사용 (2026-09-19)

- **목적**: `OPENAI_BASE_URL`로 가리킨 LiteLLM 같은 OpenAI 호환 게이트웨이가 서빙하는 모델을 `openai:<이름>`으로 고르고 기본값으로 쓸 수 있게 합니다. upstream은 `agent/dashboard/options.py`의 고정 목록만 허용하고, 직접 OpenAI 경로에는 Responses API를 강제했습니다.
- **진입점**
  - `agent/dashboard/options.py`: 기존 목록을 `BUILTIN_MODELS`로 두고 `OPEN_SWE_EXTRA_MODELS_FILE`의 JSON(`load_extra_models`)을 합쳐 `SUPPORTED_MODELS`를 만듭니다. 파일 항목은 새 모델을 추가하거나, 내장 모델의 필드를 덮어쓰거나, `hidden`으로 내장 모델을 숨깁니다. `context_window`는 프로필 오버라이드에 병합되고, 숨겨진 기본 모델은 `_default_model_id`가 같은 provider의 다음 모델로 대체합니다.
  - `agent/utils/model.py`: `OPENAI_USE_RESPONSES_API=false`면 직접 OpenAI 경로도 Chat Completions로 호출하는 `openai_use_responses_api`. `fallback_model_id_for`는 추가 모델에는 하드코딩된 타사 폴백을 적용하지 않고, 폴백 대상이 숨겨졌으면 폴백을 끕니다.
  - `agent/dashboard/workspace_settings.py`: 스레드 제목 모델의 하드코딩 기본값도 `_resolve_default_pair`를 거쳐 숨겨진 모델을 피합니다.
  - `agent/config.py`: 환경변수 `OPEN_SWE_EXTRA_MODELS_FILE`, `OPENAI_USE_RESPONSES_API`
- **설정** (`docs/INSTALLATION.md` 4절, `docs/CUSTOMIZATION.md` 2절, 예시 `examples/extra-models.json`)
  - `OPENAI_BASE_URL`, `OPENAI_API_KEY`(게이트웨이 키), `OPEN_SWE_EXTRA_MODELS_FILE`, 그리고 `LLM_MODEL_ID`/`LLM_FALLBACK_MODEL_ID`를 `openai:` 추가 모델로 지정
  - 게이트웨이가 `/v1/responses`를 못 받으면 `OPENAI_USE_RESPONSES_API=false`
- **설계상 선택**
  - 모델 파일은 기동 시 한 번 읽습니다. 잘못된 항목은 파일과 항목 번호를 담은 `ValueError`로 기동을 멈춥니다. 조용히 빠뜨리는 것보다 낫습니다.
  - effort는 같은 provider의 내장 모델이 제공하는 값만 허용합니다. `openai:`에 `minimal`을 주면 요청에서 조용히 빠지기 때문입니다. `openai:` 모델은 effort를 OpenAI `reasoning`(Chat Completions면 `reasoning_effort`)으로 보내고, 백엔드별 매핑과 미지원 파라미터 제거(`drop_params`)는 LiteLLM에 맡깁니다.
- **충돌 위험**: `options.py`의 모델 목록과 `model.py`의 `make_model` OpenAI 분기는 upstream이 자주 바꿉니다. 목록 이름을 `BUILTIN_MODELS`로 바꾼 부분은 upstream이 모델을 추가할 때마다 충돌하지만 해결은 기계적입니다.
- **upstream 가능성**: 중간. 외부 파일로 모델 목록을 넓히는 것과 Responses API 토글은 범용적이라 PR 후보입니다.
- **테스트**: `tests/dashboard/test_extra_models.py`, `tests/sandbox/test_gateway.py`(토글), `tests/models/test_model_fallback_resolution.py`(폴백)
