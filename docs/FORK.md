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
