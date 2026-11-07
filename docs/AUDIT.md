# Implementation audit — CPU-verified v0.1

Environment: container CPU only, PyTorch 2.10.0+cpu. No Modal GPU calls.
Reviewer: author self-review, not an independent reviewer.

## Checkpoint 1: model and data
- 20 CPU tests pass: exact 44,577,280 meta count; causal future invariance;
  cached single/multi-token equivalence; GQA vs repeated-KV reference;
  finite gradients; reversible UTF-8 BPE; train-only tokenizer; grouped split;
  response-only masks; immutable corpus hashes; deterministic stream restoration.
- Corrected the corruption-test fixture: 0x0101 was the unchanged BOS=257
  encoding, so it did not corrupt the file. The test now changes it to 0x0000.
- No corpus, scientific claims or pretrained weights were used in these tests.
- Network clone/install unavailable: implementation is isolated locally and will
  be published through the connected GitHub action. Reference BPE uses stdlib.

## Checkpoint 2: actual CPU training
- 33 tests pass. A tiny fixture model's teacher-forced loss decreases by >35%.
- Eight uninterrupted updates and 3+5 interrupted/resumed CPU updates produce
  bitwise-equal model weights AND Adam states, including the context transition.
- Token-aware replay, supervised response masks, total token cap, checkpoint
  corruption refusal and parent immutability pass. Fixtures are not scientific data.
- Financial guard is a time estimate with reserve, NOT an account billing cap.
  Provider budget settings remain mandatory; no charged execution was performed.

## Final review and regression fixes

Reviewed model, tokenization, data, trainer, CLI and cloud lifecycle against the approved decision.
No independent agent/reviewer was available. This is an author self-review.

Findings reproduced as failing tests, then fixed:
1. Supervised examples could be split away from their question. Now oversized
   SFT/correction records are rejected at window construction.
2. A nonempty string could masquerade as evidence. Evidence must be a list of
   paper_id/version/span records; this validates provenance shape, not scientific truth.
3. NaN RMSNorm epsilon was not rejected. Numeric validation now rejects it.
4. Last Volume persistence callback preceded lock removal. Final callback now
   runs after cleanup, so normal termination does not persist a stale run lock.
5. Parent holdout provenance was not carried into updates. Paper IDs, grouped IDs
   and exact-content hashes now propagate; detected overlap with training/replay fails.
6. Added an exact new-target-token limit, so replay experiments can match NEW
   exposure rather than accidentally matching total update count.
7. Added an explicit local cooldown transition preserving Adam/RNG/counters.

Final local suite: **50 passed** (CPU). Includes actual CLI smoke, cached GQA,
finite gradients, SHA corruption checks, response masks, variable padding,
bitwise checkpoint/resume and failure-path checks.

Full-size functional check: 44,577,280 parameters; CPU FP32; batch 1; sequence 16;
2 threads; PyTorch 2.10.0+cpu; Python 3.13.5. Forward, backward and one Adam step
completed with finite gradients and changed weights. This is NOT a throughput
benchmark and used random software-test inputs, not scientific papers.

Tiny synthetic end-to-end smoke: mean teacher-forced loss 5.7484 -> 3.0330 on the
fixture training text; 8+8 resume matches a 16-step uninterrupted reference.
The generated text is not useful scientific writing. No scientific-quality claim.

## Remaining limits / decisions

- No charged Modal run and no CUDA/BF16 test. Cloud image/API definitions were
  checked against official docs, not executed. CI is not assumed green until observed.
- HF/PyArrow sample fetching requires optional packages/network; not executed here.
- Stdlib BPE avoids unavailable installation dependencies; CPU preprocessing may
  dominate on large inputs. Benchmark before using the entire data budget.
- Replay ratio is token-accounted with bounded example-level fluctuation; actual
  new/replay counts are reported. It is not exactly 20% in every short update.
- Scientific correctness, novelty, grounding and citation checks need a separately
  curated evaluator. `verified` fields are human attestations, not model proof.
- Near-duplicate contamination is not automatically solved by exact hashes/group IDs.
- No license grant selected. No raw corpus, weights or credentials will be published.

## Official implementation references (checked 2026-11-06)

- PyTorch SDPA / GQA: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- PyTorch 2.10 CUDA 12.8 wheel: https://pytorch.org/get-started/previous-versions/
- Modal Volumes and commit: https://modal.com/docs/guide/volumes
- Modal App/Function: https://modal.com/docs/reference/modal.App
- Modal Image: https://modal.com/docs/reference/modal.Image
- Modal timeout: https://modal.com/docs/guide/timeouts
- Provider budget: https://modal.com/docs/guide/budgets

## Publication verification

- Implementation commit: `93a3a7d8d0e9b231871de5e7303fc02abddeb52a`.
- Feature branch: `feat/trainable-map44m`; PR: https://github.com/Unjuno/model-as-a-paper/pull/1
- Implementation tree `b96c666706192c1b4be511991aee1a00d3080399` matched the tested local tree exactly.
- 30 public text files scanned; no raw corpus, checkpoints, private-key material or credentials detected.
- Editable package installation and wheel build also succeeded offline with installed dependencies.
- `map-paper inspect` reports 44,577,280 parameters through the installed console entrypoint.
- GitHub CPU CI was triggered; consult the PR checks for its current status. No automatic merge.

## 2026-11-08: 学習前準備・実行ノブ・計測の追加監査

- 変更後の**70 CPU tests passed**。PyTorch 2.10.0+cpu / Python 3.13.5。
- `preflight`: CorpusのSHA/manifest、実使用語彙、非空train windows、モデル構造、
  parentのholdout保護、Replay Tokenizer、test fixture混入をGPU確保前に確認。
- `bench`: モデルの実勾配更新を計時し、計測値を報告。学習checkpointの保存なし。
- `stats`: 実行ログと最後の確定checkpointの整合を確認。
- `--micro-batch-size`, `--checkpoint-steps`, `--checkpoint-seconds`: 再開で変更できるruntime knob。
  保存された学習目的／データ・モデル構造は変更しない。
- 再開の未確定ログ: 意図的にstep4の偽ログを追記して重複をRED再現し、
  `metrics.orphaned-*.jsonl`へ退避してから継続することでGREENへ。
- `setup_training.py`: 明示的なJSONLまたはHFの小サンプルで準備→preflightを1操作に集約。
  HTTP取込時のstdoutでJSON結果が破壊される問題をRED再現して修正。
- `fetch_arxiv_papers.py`: 70GB全量を取らないよう、パス・revisionを固定し、
  remote Parquet row groupの**推定**スキャンbytes、出力bytes、論文件数で限定。
  この経路の実ネットワーク挙動はまだ検証していない。
- Modal: runtime knobと`start_cooldown`を公開し、保存済みcooldownだけを許す
  resume設定照合を追加。Cloud image/API・CUDA上の実行テストは未実施。
- 全44M本体のCPU FP32 forward/backward/Adamを再実行：有限勾配、重み変更、
  パラメータ数44,577,280を確認。バッチ1、系列16、2スレッド。
- コード形状・構文検査、`git diff --check`、代表的な秘密情報形式の静的スキャンを実施。

この監査は実装者自身によるもの。独立第三者監査や科学QA評価を意味しない。
更新前のチェックポイントの保護とデータのライセンス・時系列的な信頼性は引き続き
学習実行前のゲートとする。外部GPUへ課金ジョブは送信していない。
