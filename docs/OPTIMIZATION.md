# GPU学習を最適化しやすいようにする運用契約（MAP-44M v0.1）

**性能改善を主張するための実GPUベンチマークは未実行。**
最初は学習内容を変えず、実行性能だけを変えられるパラメータを測る。

## 1. 変えてよいもの／変更すると新しい実験になるもの

| 種類 | 設定 | resume時の扱い | 品質比較での扱い |
|---|---|---|---|
| 実行用 | `--micro-batch-size` | 可 | 有効batch token数は同一。数値丸め差はあり得る |
| 実行用 | `--checkpoint-steps` / `--checkpoint-seconds` | 可 | GPU時間に保存時間を含めて比較する |
| 計算単位 | `training.batch_tokens` | 不可 | 変えるなら別run/実験 |
| モデル | 層数、幅、head数、語彙、RoPE | 不可 | 新モデル＋再学習。名前を変える |
| 学習 | LR、warmup、replay比、乱数、系列長 | 不可 | 新run/新目的。ただし専用cooldown遷移あり |
| コーパス | manifest、tokenizer、split | 不可 | 通常resumeは拒否。別runを作る |

## 2. 学習前に必ず

```bash
python -m pytest -q
python scripts/setup_training.py --output data/papers --max-papers 256
python -m model_as_a_paper.cli preflight --config configs/map44.json --data data/papers
```

長期学習に256論文のサンプルを使い続けない。実際の主学習は、十分な分量の論文本文を
`--source paper-text`または権利・版を確認した独自JSONLで準備する。
最初からHFの全16TBを取得しない。

## 3. Modalでの最小試験と計測

Modal公式の現在価格・Workspace上限・残クレジットをUIで確認する。
時給はGPU+CPU+メモリの**総単価**を入力する。`--budget-usd`は推定計算時間の制限であり、
クラウド提供者の硬い請求上限ではない。

```bash
modal volume create map-data
modal volume put map-data data/papers /corpora/papers
modal run modal_apps/train.py --dataset papers --run-name pretrain-001 \
  --config configs/map44.json --steps 20 --seconds 300 \
  --budget-usd 0.20 --hourly-rate 1.10 \
  --micro-batch-size 2 --checkpoint-steps 5 --confirm-cost
modal volume get map-data /runs/pretrain-001 runs/pretrain-001
python -m model_as_a_paper.cli stats --run runs/pretrain-001
```

再開時には、同じ`run-name`と`--resume`を指定する。
GPUメモリが厳しい場合は**runtime only**の`--micro-batch-size 1`に変更する。
通常はlossのlogを比較する前に、実際の更新トークン数・使用データ・学習設定を確認する。

## 4. 速度測定の条件

- 同じGPU型番・台数・CUDA/PyTorch/SDPA backend・BF16精度・電力制限／クロック（取得可能なら）。
- 同じ原文、モデル、系列長、有効batch tokens、微調整目的。Microbatchだけを変える試験ではその旨を明記。
- 準備・Image build・保存・評価・転送をGPU step時間と**別記**し、最終判断では全費用を合算。
- 先に数stepでメモリ・実行速度を測り、複数stepの中央値と範囲を比較。
- `torch.compile`や別kernelの導入は、正答・数値の回帰を確認後、別ブランチで実施。
- H-Netとの比較では同じ原文byte数、同じFLOPs、同じ課金額を別々に比較する。

## 5. 測定結果からの判断

`metrics.jsonl`は`step_seconds`（CUDA完了まで同期した実時間）、`input_tokens`、
`cuda_peak_memory_mib`、`runtime_micro_batch_size`、`loss`を記録する。
`stats`は最新checkpointより先の未確定ログを拒否する。強制停止などで先行ログが
残った場合、`resume`で監査ファイルへ退避してから次stepを記録する。

**H-Net・Mamba・ESは初期ブランチに混ぜない。**
設計上の変更は、まず独立branchでBPB/科学QA/更新後の保持と実課金まで比較する。

## 6. 残る制限

- Modal SDK・CUDA/BF16・実GPU・HF遠隔ParquetはこのCPU環境で未検証。
- 自動的なOOMリトライ/自動microbatch探索は未実装（optimizerの途中状態を汚染するリスク）。
- 標準ライブラリのBPEはCPU参照実装で、実コーパス前処理速度の最適化対象。
- `paper_text`のバージョンは追跡されない。訂正の厳密な時系列評価には使わない。
- 科学的正確性・References実在性・真の忘却は専用の独立評価器が必要。
