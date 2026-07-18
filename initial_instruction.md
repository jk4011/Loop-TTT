# Initial Instruction (2026-07-18, 사용자 원문 정리)

> 이 파일은 프로젝트 시작 시 사용자가 준 지시사항의 정리본이다. context가 compact되어도 이 파일을 기준으로 작업을 이어간다.

## Task Introduction

- **TTT**(Test-Time Training layer)는 attention을 대체하는 linear-time 방법론. quadratic한 Transformer의 대안으로 주목받음.
- **Looped Transformer**: 하나의 layer를 여러 번 반복(weight-tied loop)하여 파라미터를 아끼면서, loop 횟수만큼의 layer를 가진 모델과 거의 유사한 성능을 냄.
- **하지만 TTT layer에서 looping이 동작하는지에 대한 분석은 아직 없다.**
- 본 연구: TTT layer에서 looping이 동작하는지 확인하고, TTT의 특성과 looping의 특성을 함께 고려한 **novel한 방법론**을 제안한다.

## Related Work — TTT

1. **TTT** (Learning to (learn at test time): RNNs with expressive hidden states)
   - key를 input으로 넣어 TTT layer(MLP)가 output을 예측 → value와의 loss로 backprop 1회 update → key→value 매핑을 학습.
   - 이후 query를 넣으면 해당하는 value가 나옴. attention의 "q·k 유사도 기반 value interpolation" motivation을 그대로 공유.
2. **LaCT** (Test-Time Training Done Right) — https://github.com/a1600012888/LaCT.git
   - Large-Chunk TTT: 한 번에 많은 token(q,k,v 쌍)으로 update → 병렬성/GPU 활용도 ↑.
   - NVS(LVSM)에 적용 시 full attention과 유사한 성능 + 몇 배 빠른 속도.
   - **우리 baseline은 LaCT 기반. baseline task는 NVS (LaCT LVSM).** 방법론이 구체화되면 다른 task로 확장.
3. **기타**: TTT에 looping을 적용한 연구가 있는지 확실히 조사할 것. Mamba+looping은 arXiv/workshop 수준 논문 존재 (신뢰도 낮음):
   - Looped SSMs: Depth-Recurrence and Input Reshaping for Time Series Classification
   - Tiny Recursive Reasoning with Mamba-2 Attention Hybrid

## Related Work — Looped Transformer

1. **이론적 토대**: Universal Transformers (ICLR'19), ALBERT (ICLR'20), Looped Transformers as Programmable Computers (ICML'23), Looped Transformers are Better at Learning Learning Algorithms (ICLR'24, arXiv:2311.12424).
2. **LLM 추론/잠재 사고**: Reasoning with Latent Thoughts (arXiv:2502.17416) — 얕은 모델 loop ≈ 동일 총깊이 비루프 모델, loop가 숨겨진 추론 단계를 시뮬레이션. Coconut (arXiv:2412.06769). Looped Transformers for Length Generalization (ICLR'25, arXiv:2409.15647).
3. **대규모 확장**: Huginn recurrent-depth 3.5B (NeurIPS'25, arXiv:2502.05171). Ouro/LoopLM (arXiv:2510.25741) — 1.4B/2.6B가 12B급 성능(2–3× 파라미터 효율). Parallel Loop Transformer (arXiv:2510.24824).
4. **소형 재귀 추론기**: HRM (arXiv:2506.21734), TRM (arXiv:2510.04871, 7M param으로 ARC-AGI-1 45%).
5. **비전**:
   - **RAPTOR** (Block Recurrent Dynamics in Vision Transformers, ICLR'26) — looping transformer를 vision으로 확장.
   - **Déjà View** (Looping Transformers for Multi-View 3D Reconstruction, arXiv:2605.30215) — looped ViT로 multi-view 3D recon. **중요**: loop가 단순 파라미터 절약이 아니라 layer 간 중복 정보를 학습하기 때문에 multi-layer보다 오히려 높은 성능 달성. 우리도 "파라미터↓ + 기존 LaCT 초과 성능"을 보이면 임팩트 큼.

이 외에도 주목할만한 최신 연구가 있으면 조사/분석할 것.

## Related Work — NVS

- **LVSM**: posed multi-view image + target view condition을 Transformer에 넣고 target view image를 출력하도록 학습 → minimal 3D inductive bias로 NVS. **우리의 target task.**

## Task Goal: LaCT에 Looping 접목

- 단순 LaCT+looping은 contribution 부족 → **LaCT+looping을 baseline으로 두고, 이를 향상시키는 novel 방법 개발이 목표.** (RAPTOR도 비교 baseline 후보.)
- TTT와 looping을 같이 쓸 때의 **핵심 challenge를 규명하고 해결**하는 novelty 있는 연구. 수학적 intuition이 있으면 좋음 (필수는 아님).
- **최대한 다양한 방향으로 ~10가지 아이디어(가설)** 생성 — subagent 브레인스토밍 + 추합 권장.
- 코딩/실험으로 가설 검증. **GPU: B200 4대 → 동시 4개 실험.**
- baseline 대비 향상 확인 → 유망하면 **seed 3개로 검증** → 새 variant 파생 → 반복 (**유전알고리즘식 방법론 진화**).
- **방법론은 미니멀하게 유지** — 이해하기 쉽고 널리 쓸 수 있게. 복잡해지는 것 방지.

## Project Structure

- `./lact` — LaCT github clone (`.git` 삭제, 단일 repo로 관리). looped TTT 구현 장소. `lact/CLAUDE.md`에 코드 구조 정리.
- `./related_work/[paper_name]/` — 참고 paper + 해당 code, 정보는 각 폴더의 `CLAUDE.md`에 정리.
- `../dataset` — 새 데이터셋 다운로드 위치.
- `./experiment_queue.md` — 실험 queue. GPU가 비는 대로 바로 실험 실행. queue 순서 유기적으로 조정/삭제.
- `./CLAUDE.md`, `./initial_instruction.md` (이 파일).

## Other Background

- Slurm batch 환경: **2일마다 만료** → `~`, `/tmp` 삭제, 프로세스 강제 종료. `./`(lustre)는 보존. **주기적 checkpoint 정리 필수.**
- 지금은 B200 4개, 나중에 8개 노드로 확장 가능.
- `/tmp`는 tmpfs(RAM) — batch 재시작 때마다 데이터를 옮겨두면 빠른 로딩 가능.
- Python: 기존 환경 활용 (`/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm`).
- **모델 크기 축소: dim 256 (768의 1/3), layers 8.** GPU 1개당 실험 ~2시간이 적당.
- **주기적으로 git commit & push.** clone받은 repo(LaCT 등)는 `.git` 삭제해서 단일 repo로 관리.
