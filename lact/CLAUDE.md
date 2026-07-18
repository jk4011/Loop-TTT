# lact/ — 코드 구조

Upstream https://github.com/a1600012888/LaCT.git 클론(.git 제거) + 이전 프로젝트
(`TTT_camera_embedding/lact_nvs`)에서 포팅한 실험 인프라. 작업 디렉토리는 **`lact_nvs/`** 하나.
`lact_llm/`, `lact_ar_video/`는 upstream 그대로(읽기 전용), `minimal_implementations/`는
LaCT layer의 교육용 standalone 구현 (TTT update 수식 이해에 최적).

## lact_nvs/ 데이터 흐름

posed multi-view 이미지 → per-pixel ray map(ray_o, ray_d, o×d) + RGB → patchify(16) → linear →
`Block` 스택 (× n_loops 반복) → target 토큰 decode → RGB.
Input view 토큰이 fast weight를 **update**, target view 토큰은 **apply**만.

## 파일별

- **`model.py`** — `LaCTLVSM` 최상위.
  - `__init__(patch_size, dim, layers, block_config, n_loops=1, ttt_state_mode="reset"|"carry", input_injection="none"|"add")`
    — **looping은 우리가 추가한 확장**. `layers` = 고유 Block 수 (loop 단위), effective depth = layers×n_loops.
    c_proj 잔차 init은 effective depth로 스케일.
  - `forward()` (훈련/eval): `for loop_idx in range(n_loops): for block in blocks` 순회.
    - `ttt_state_mode="carry"`: block이 반환한 `{w0,w1,w2}`를 `block_states[block_idx]`에 저장해
      다음 loop pass의 info로 주입 → fast weight가 loop 간 누적 update (block별 상태 분리 주의).
    - `"reset"`: 상태 버림 → 매 pass 학습된 init에서 재시작 (naive/attention식 loop).
    - `input_injection="add"`: 매 loop 시작 시 embedded input(x0)을 재주입 (Huginn/Looped-TF 관례).
  - `reconstruct()`/`rendering()` (추론): states를 **방문 순서**(n_loops×layers 항목)로 저장/재생.
  - `Block` = YAML `block_config`대로 [ln→f(SelfAttention/TTT/MLP)→residual] 서브모듈 스택.
    `length_dim: "l"`=view 내부, `"vl"`=전체 view 토큰 (TTT/MLP가 여기; cross-view는 TTT에서만 일어남).
- **`lact_ttt.py`** — LaCT TTT layer (**baseline, 수정 금지** — 새 방법은 별도 파일로).
  - `FastWeightGluMLPMultihead`: fast weights w0,w1,w2 = per-head SwiGLU. q,k,v는 silu(to_qkv),
    q/k L2-norm. per-token lr = softplus(lr_fc(x)+base_lr_inv). `info["w0"]` 있으면 그걸 init으로
    사용(carry/rendering 훅), 없으면 self.w0.
  - `fast_weight_swish_glu_weight_norm_mini_batch_apply`: `ttt_op_order`의 각
    `TTTOperator(start,end,update,apply)` 세그먼트에 대해 (update면) manual SwiGLU backward →
    **`zeropower_via_newtonschulz5`(Muon; steps=0이어도 Frobenius normalize → gradient 크기 소실!)**
    → w += grad → per-column weight-norm 재정규화. (apply면) SwiGLU(q) 출력.
- **`train.py`** — 포팅된 인자: `--data_path --dataset re10k --num_workers --seed`, it/s 로깅.
  `--actckpt` 필수(Block 단위 non-reentrant ckpt; loop 반복 호출과 호환). outputs/<exp>에서 auto-resume.
- **`eval.py`** — 표준 평가: 256 held-out RE10K scene, 8in/4tgt, `forward()` 경로로 PSNR/LPIPS
  → eval.json (per-scene 배열 포함 → paired 비교 가능).
- **`data_re10k.py`** — /tmp/re10k/{train,test}_index.json 로더 (reshard된 per-scene .torch).
- **`data_preprocess/reshard_re10k.py`** — pixelsplat식 chunk → per-scene 파일 + index.json (/tmp에).
- **`launch_exp.sh`** — 표준 30k 훈련 (triton/inductor 캐시 lustre로 export — /tmp noexec 때문).
- **`chain_run.sh`** — 훈련→eval 체인. **실험은 이걸로 실행**: `chain_run.sh <gpu> <exp> <config> [seed]`.
- **`config/`** — `lact_l{2,6,8}_d256_p16.yaml` 비루프, `loop_l2x4[_carry]_d256_p16.yaml` 루프.
  네이밍: `loop_l<layers>x<n_loops>[_변형]_d256_p16.yaml`.

## Gotchas (이전 프로젝트에서 검증)

- ray/pose 전처리는 fp32(`autocast(enabled=False)`) — 새 pose 관련 수식도 fp32 유지.
- `--compile`은 첫 스텝 ~2분 컴파일; 디버깅 시 생략. `@torch.compile` 데코레이터는 항상 적용됨.
- checkpoint 로드 시 `remove_module_prefix`가 module./_orig_mod./_checkpoint_wrapped_module. 제거.
- bf16 autocast 훈련; carry되는 fast-weight 상태도 이 경로를 그대로 탐.
