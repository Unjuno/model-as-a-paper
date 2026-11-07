# Model-as-a-Paper — MAP-44M v0.1

論文だけを原資料としてスクラッチ学習し、検証済みの訂正をモデルの重みに学習するための実装です。
**学習器であり、科学的能力が確認された学習済みモデルではありません。**

44,577,280 parameters / 12 layers / width 512 / GQA 8:2 / SwiGLU 1536 /
16,384 vocabulary capacity / tied embedding / RoPE / PyTorch SDPA.
H-Net・ES・RAG・一般事前学習済み重みは初版に含めません。

## 検証状態

- CPU: 70 tests passed（2026-11-08、ローカル再検証）。実際の小型モデル学習、保存・再開、訂正学習、Replayを検証。
- MAP-44M本体: CPUでforward/backward/Adam更新を実行、有限勾配と重み変更を確認。
- 中断なし8更新と3+5更新: CPU上で全重みとAdam状態がbitwise一致。
- **未検証:** Modal image build、CUDA/BF16、GPU速度・費用、HF遠隔の実論文取得処理、科学的回答品質。
- 監査は実装者自身の別パスのレビュー。独立第三者レビューではありません。

詳細は [監査記録](docs/AUDIT.md)、[設計決定](docs/DECISION.md)、
[データ契約](docs/DATA.md)、[運用手順](docs/RUNBOOK.md) を参照。

## 1. ローカル導入（Linux / Python 3.11以上）

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CPU用。ローカルCUDA使用時は公式の対応wheelを選ぶ。
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]'
python -m pytest -q
python -m model_as_a_paper.cli inspect --config configs/map44.json
```

CUDA用はPyTorch 2.10.0 / CUDA 12.8 wheelをModal imageで指定しています。
ローカルの実測環境はPython 3.13.5 / PyTorch 2.10.0+cpuです。
GPU対応とCPU試験は別の検証です。

## 2. 課金なしの動作確認

```bash
python -m model_as_a_paper.cli smoke --output runs/smoke-001
```

人工的な**ソフトウェアテスト専用**文章で、準備→学習→再開→訂正＋Replay→生成を通します。
出力は `test_only: true` と記録され、本学習では明示的な許可なしに使用できません。
損失が下がっても科学的能力の証拠ではありません。既存出力は上書きしません。

## 3. 実際の論文データを準備 — 1コマンド

必要なJSONL形式は [DATA.md](docs/DATA.md)。CC BY 4.0 / CC0を既定の選別条件とし、
原文・出典・ライセンス・元データのrevisionを記録します。`paper_text`は**arXivの各版に対応しない**ため、
この本文だけで訂正前後の真偽を主張しません。利用権の判断を自動化するものではありません。

```bash
python -m pip install -e '.[hub]'
# 最初は26 MB程度のサンプルだけでCPU data-prep→preflightを行う。GPUは起動しない。
python scripts/setup_training.py --output data/papers --max-papers 256
```

より多くの論文が必要な場合、全70 GBの`paper_text`をダウンロードせず、
**remote Parquetの一部row group**から本文を抽出するモードを用意しました。
スキャン上限はParquetの推定圧縮bytesであり、実HTTP転送量の厳密な上限ではありません。

```bash
python scripts/setup_training.py --source paper-text --output data/papers-math \
  --max-papers 4096 --max-shards 3 --max-scan-mib 256 \
  --max-output-mib 192 --category-prefix math.
```

既存の論文JSONLを使用する場合は、`--input data/my-papers.jsonl`を指定できます。
**どの経路も自動ではクラウドGPUを起動しません。** Hugging Faceの大きなParquet遠隔読出しは
このCPU環境では未実測のため、必要に応じて`--max-scan-mib`を下げて試してください。

```bash
python -m model_as_a_paper.cli preflight --config configs/map44.json --data data/papers
```

Tokenizerはtrain splitだけで作成し、UTF-8・数式・空白を可逆に保持します。
標準ライブラリ製byte-BPEを同梱しています。速度最適化済みのRust実装ではありません。
Tokenizer学習は既定8MiB、入力JSONLは256MiBまで。大きいデータは選別・分割して扱ってください。
語彙容量は16,384ですが、小さい資料では実使用語彙がそれ未満になるためmanifestに両方を記録します。
サンプルを何百回も繰り返しても、本格的な科学コーパスの代わりにはなりません。

## 4. 事前計測・ローカル学習・再開

```bash
# GPUマシン上で、小さなforward/backward/optimizer実測。checkpointを書きません。
python -m model_as_a_paper.cli bench --config configs/map44.json \
  --data data/papers --device cuda --steps 2 \
  --micro-batch-size 1 --batch-tokens 1024
```

ローカルにGPUがない場合、CUDA benchをCPUで代用してGPU速度と呼ばないでください。
Modalでは後述の短い有料runの`metrics.jsonl`がGPU実測になります。

```bash
python -m model_as_a_paper.cli train --config configs/map44.json \
  --data data/papers --output runs/pretrain-001 --device cuda \
  --steps 100 --seconds 600 --token-limit 500000000

python -m model_as_a_paper.cli train --config configs/map44.json \
  --data data/papers --output runs/pretrain-001 --device cuda \
  --resume --steps 100 --seconds 600 --token-limit 500000000
```

`--steps`は今回の追加更新回数。`--token-limit`は当該目的の累積予測トークン上限です。
`--seconds`は保存余裕を含む要求時間で、学習の停止は更新境界です。
CPUでも `--device cpu` で実行できますが、500MトークンのCPU学習を推奨する意味ではありません。
中断はwarmupをリセットしません。`--resume`ではモデル・学習設定・データ・Tokenizerの変更を拒否します。
一方、**計算手順だけの調整**なら`--micro-batch-size 4`、`--checkpoint-steps 20`、
`--checkpoint-seconds 60`を付けて再開できます（有効batchの予測対象トークン数は不変）。
浮動小数点の加算順序が変わるため、変更後のbitwise同一性は保証しません。
停止直前の未保存ステップのログは、resume時に`metrics.orphaned-*.jsonl`へ監査保存して整理します。

`python -m model_as_a_paper.cli stats --run runs/pretrain-001`でcheckpointとログを照合し、
step時間・最新損失・処理トークン数を確認できます。詳細：[最適化手順](docs/OPTIMIZATION.md)。

## 5. Modalで明示的に起動する

**ここからは課金され得ます。** 実クレジットとWorkspaceの予算上限を先に確認してください。
時間からの費用推定は、画像構築・保存・実価格変動・他ジョブを含む請求の上限保証ではありません。

```bash
python -m pip install -e '.[cloud]'
modal setup
modal volume create map-data
modal volume put map-data data/papers /corpora/papers

# 初回は短く動作確認。1.10 USD/hは設定例で、実料金を確認して置き換える。
modal run modal_apps/train.py --dataset papers --run-name pretrain-001 \
  --config configs/map44.json --steps 20 --seconds 300 \
  --budget-usd 0.20 --hourly-rate 1.10 \
  --micro-batch-size 2 --checkpoint-steps 5 --confirm-cost

# 同じデータ・設定・run名を指定して再開。
modal run modal_apps/train.py --dataset papers --run-name pretrain-001 \
  --config configs/map44.json --steps 100 --seconds 600 \
  --budget-usd 0.50 --hourly-rate 1.10 --resume --confirm-cost
```

L4×1 / CPU 2 / RAM 8192MiB / 最大1 container / 自動retry 0 / hard timeout 3600秒。
実測したGPUメモリ・step秒数を見てmicrobatchを変更できます。学習目的・Tokenizerを変更する場合は
resumeではなく別のrunを作成します。cooldownは`--resume --start-cooldown`で開始し、
その後は同じチェックアウトの構成を渡しても保存済みcooldown開始点だけを自動反映します。
要求する学習時間は最大3000秒で制限し、長い学習は状態を保存して再開します。
`--confirm-cost`がなくてもSDK側のimage準備が先に走る可能性があるため、
認証なし・課金なしで試す場合はModalではなくローカル`smoke`を使ってください。

## 6. 論文由来SFT・訂正学習

学習用の `prompt` は科学的質問、`answer` は根拠・条件・限界を持つ論文形式の教師文です。
実験や引用を捏造する自動生成器は含めていません。`verified`は監査記録であり、真理判定ではありません。

```bash
python -m model_as_a_paper.cli prepare --input data/corrections.jsonl \
  --output data/corrections --tokenizer data/papers/tokenizer.json \
  --split train --min-chars 1

# replay用データは、既知の誤説を無条件の正解として含まないよう別途選別する。
python -m model_as_a_paper.cli train --config configs/update.json \
  --data data/corrections --replay data/replay-curated --output runs/update-001 \
  --init-from runs/pretrain-001 --device cuda --steps 100 --seconds 600 \
  --new-token-limit 100000
```

Replayも親と同じTokenizerで準備してください。20%は予測対象トークンで追跡する目標比率です。
例の長さにより短い実行の実比率はずれます。新規・Replayの実トークン数をログに記録します。
比較は同じ`--new-token-limit`で行い、総トークン・費用の増分を別に計上します。

`--init-from`は親の重みとデータ由来情報を引き継ぎ、新しい目的のoptimizer・カウンタを開始します。
`--resume`は同じ目的を厳密に継続します。両者を混同しないでください。
親モデルが保持する評価用論文ID・group・内容hashが追加学習へ入ると拒否します。
近重複や意味的な漏洩は別途人手で確認が必要です。

## 7. 評価と生成

Modalの結果をローカルへ取得して実行できます。

```bash
modal volume get map-data /runs/pretrain-001 runs/pretrain-001
python -m model_as_a_paper.cli evaluate --checkpoint runs/pretrain-001 \
  --data data/papers --split validation --seq-len 1024 --max-windows 100
python -m model_as_a_paper.cli generate --checkpoint runs/update-001 \
  --prompt 'Under which assumptions does the corrected claim hold?' --max-new-tokens 256
```

評価器はNLL・BPBを出します。**科学QA正答率・引用支持率・忘却耐性を自動保証しません。**
評価集合が空の場合は失敗し、train集合へ黙って置き換えません。
生成は未検証である旨を付し、コンテキスト超過の切り捨てをしません。
回答のReferencesは別途実在・主張対応を確認し、確認できないものを成果として公開しないでください。

## 構成

- `src/model_as_a_paper/`: model、byte-BPE、データ、trainer、評価・生成、CLI。
- `configs/`: 基盤学習と訂正更新の設定。
- `scripts/`: 1コマンド準備、サンプル／選別本文の限定取得、本体44MのCPU検証。
- `src/model_as_a_paper/readiness.py`: 学習前診断、実測bench、checkpoint付き統計。
- `docs/OPTIMIZATION.md`: 実GPUでの比較条件と安全な性能調整手順。
- `modal_apps/train.py`: 課金確認付きの単発・保存再開型GPU起動。
- `tests/`: 因果性、更新、データ漏洩、再開、費用境界の回帰試験。
- `.github/workflows/tests.yml`: CPUテストのみ。クラウド資格情報・GPUジョブなし。

## 公開とライセンス

コードのライセンスはまだ選択されていません。[NOTICE.md](NOTICE.md)を参照。
論文・モデル重み・秘密鍵・トークンはcommitしないでください。
