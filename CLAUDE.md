# CLAUDE.md — Loop-TTT 프로젝트

## 연구 목표

**TTT(test-time-training) layer에서 looping(weight-tied depth recurrence)이 동작하는지 규명하고,
TTT의 stateful 특성 × looping의 특성을 이용한 novel 방법론을 제안한다.**
Baseline: LaCT (Large-Chunk TTT) + LVSM-style NVS, RE10K. 원 지시사항: `initial_instruction.md`.

핵심 통찰 (자세히: `IDEAS.md`): attention은 loop에 stateless하지만 TTT fast weight는 loop마다
재update 가능 → looping = inner-loop epoch 추가. 단, LaCT update는 Muon/NS가 gradient 크기를
지워 residual 무관 고정 스텝 → 반복 시 수렴 대신 궤도 순환이 예상되는 것이 core challenge.

**Novelty 확인됨** (related_work/SURVEY.md): nonlinear TTT layer의 looping 연구는 전무.
최근접: LT2(2605.20670, linear mixer 한정, LM), Déjà View(2605.30215, attention ViT, NVS).

## 저장소 구조

- `lact/` — LaCT upstream clone(.git 제거) + 이전 프로젝트에서 포팅한 인프라. **작업 장소는 `lact/lact_nvs/`.**
  코드 구조: `lact/CLAUDE.md` 참조.
- `related_work/` — SURVEY.md + 논문별 폴더(paper + code + CLAUDE.md).
- `IDEAS.md` — 가설 목록(I1~I12) + 중심 이론. `experiment_queue.md` — 실험 큐/실행 중/완료.
- `RESULTS.md` — 모든 eval 결과 + baseline 대비 paired Δ (진실의 원천).
- `initial_instruction.md` — 사용자 원 지시사항 정리본.
- `../dataset/` — 새 데이터셋 다운로드 위치 (RE10K 원본은 `/NHNHOME/WORKSPACE/26msit001_A/V-LAB/Datasets/re10k`).

## 실험 프로토콜 (이전 TTT_camera_embedding 프로젝트에서 검증된 것 계승)

- 표준 런: `bash lact/lact_nvs/chain_run.sh <gpu> <exp> <config> [seed]`
  = 30k iters, bs16, lr1e-4, LPIPS from 5k, RE10K 256², 8in+8tgt → 256-scene eval → eval.json.
- 소형 모델: dim 256, effective depth 8 (L8 비루프 또는 L2×4루프 등), patch 16. 런당 ~2.2h/B200.
- **seed-matched 비교 필수**: 단일 seed ΔPSNR ~0.1dB는 노이즈. 유망하면 seed 95/96/97.
- GPU 4개 (나중에 8개로 확장 가능). GPU 점유는 `lact/lact_nvs/outputs/.gpu_locks/gpu<i>`에 기록.
- 실험 결과/큐 갱신 → git commit & push 주기적으로.

## 환경 (ephemeral Slurm 노드 — 상세는 ~/.claude/CLAUDE.md)

- 2일마다 노드 리셋: `~`, `/tmp` 소멸, 프로세스 강제종료. lustre(`./`)만 영구.
- 리셋 후: (1) `bash /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/claude_portable/setup_node.sh`,
  (2) RE10K reshard: `lact/lact_nvs/data_preprocess/reshard_re10k.py --src .../re10k/{train,test} → /tmp/re10k`
  (train+test 각각, ~3분), (3) experiment_queue.md 보고 미완료 런 재개 (train.py는 outputs/<exp>에서 auto-resume).
- Python: `/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm` (torch 2.11+cu128, B200 대응). 새로 만들지 말 것.
- `/tmp`는 noexec → triton/inductor 캐시는 lustre로 (launch_exp.sh가 export; 새 런처 만들 때 복사할 것).
- checkpoint: 최종본만 유지하고 중간 것 삭제. outputs/는 lustre라 영구.

## 용어 (LaCT 논문 따름)

fast-weight gradient step = **update** (not "write" in code/docs), fast weights를 query에 사용 = **apply**.
단, 우리 방법 서사에서 write/read 비유를 쓸 때는 명시적으로 (I8처럼).

## 2-노드 운영 (2026-07-22~)

- 노드가 2개일 수 있음(주 노드 B200×4 + 보조 노드 B200×6). lustre 공유.
- **작업 큐 = `NODE_QUEUE.md`** (프로토콜 문서화됨). 주 노드 Claude가 설계·큐잉, 보조 노드
  Claude가 claim→실행→기록. 실험명 노드 간 겹침 금지, 커밋 전 `git pull --rebase` 필수.
- 보조 노드에서 이 폴더로 claude를 열면: 먼저 NODE_QUEUE.md를 읽고 프로토콜대로 진행할 것.
  NVS 실험 전 /tmp/re10k reshard (NODE_QUEUE.md 준비 절 참조).
