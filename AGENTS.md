# AGENTS.md

> **시작:** [공통 WORKFLOW_GUIDE v1.7](https://github.com/berandaNuguri/berandaNuguri/blob/main/WORKFLOW_GUIDE.md) → README → 아래 추가 규칙.
> 공통 업로드·문서 정책은 GUIDE를 따른다. 아래는 **JPEG-aware 프로젝트**의 적용 기준이다.

## 1. 파일 위치

```text
src/
├── data/                       # 실제 데이터: Git 제외
└── jpeg_aware_steganalysis/
    ├── models/                 # 모델 구현
    └── utils/data_processing/  # 준비·로더 코드
scripts/                        # 작업별 실행 진입점
checkpoints/                    # 모델 파일: Git 제외
```

## 2. 보존 / 금지

| 구분 | 규칙 |
| --- | --- |
| 변경 전 | README와 관련 import·entry point·리소스 경로 확인 |
| 동작 보존 | 모델 계산·전처리·split·seed·CLI 기본값·checkpoint state_dict key 유지 |
| 원본 프로젝트 | 이 checkout 바깥의 원본 paper 프로젝트 수정 금지 |
| 소스 구조 | 실행 소스는 src/jpeg_aware_steganalysis/, 진입점은 scripts/ |
| 이전 호환 | 호환 symlink·legacy 최상위 arch/·utils/ 패키지 추가 금지 |
| Git 제외 | 데이터·CSV·그림·checkpoint·결과·로그·논문 원고 |
| W&B | 로컬 Run·계정 설정·binary history·인증정보 commit 금지; shell/비공개 외부 credential store로 인증 |

## 3. 경로 설정

| 항목 | 기준 |
| --- | --- |
| CLI 상대 경로 | 이 checkout root 기준; 호출 shell의 cwd와 무관하게 해석 |
| 데이터 / 모델 기본 경로 | Git 제외 src/data/ / checkpoints/ |
| 외부 경로 | RESEARCH_DATA_ROOT / RESEARCH_MODEL_ROOT |
| 과거 input prefix | 비공개 RESEARCH_PATH_MAP JSON 환경변수 |
| 금지 | host 경로·컨테이너 이름 hardcode |

## 4. README / 완료 전 확인

| 대상 | 기준 |
| --- | --- |
| 첫 내용 | 연구 문제·확인된 논문 정보·방법부터 설명 |
| 출판 정보 | 논문·1차 출판 기록으로 검증; 폴더명·manuscript template으로 게재·venue·연도 추정 금지 |
| 운영 기록 | 이전 이력·검증 로그는 docs/ 또는 AGENTS |
| 검증 | 프로젝트별 지침 먼저 확인; 실제 수행한 검사와 범위만 보고 |

- [ ] 관련 경로·import와 기존 동작 보존 확인.
- [ ] staged 파일에서 데이터·가중치·결과·개인 값 제외.
- [ ] commit/push 후 remote 반영 확인.
