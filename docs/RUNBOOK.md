# 運用・監査Runbook

## 最初の順序
1. CPU testsとsmokeを通す。人工fixtureが本学習に混ざっていないことを確認。
2. 版・利用条件を確認した小さな論文集合をprepareする。
3. `manifest.json`の非空split、実語彙数、重複groupを確認する。
4. Modal Workspace予算・実クレジット・実リソース料金を確認。
5. L4の短い単発実行。loss、有限勾配、保存・再開、実測費用を確認する。
6. 実際のGPU速度が許容できてから実行量を増やす。科学的品質は別ゲート。

## 保存と再開

`latest.json`はimmutable checkpointファイル名とhashを持つatomic pointerです。
1つ前のpointerは`previous.json`。自動保持は最新2ファイルです。
重要な親モデル・良好な評価結果は別runへコピーして保管してください。
全失敗時に無条件でcheckpointを保存する設計ではなく、有限勾配の更新を完了した時だけ保存します。
非finiteや異常終了では最後に正常保存したcheckpointを使います。

runの`.running`は競合検出用です。正常終了と制御された例外では解除しVolumeへcommitします。
OS停止・強制killでは残る可能性があります。必ず該当ジョブが終了していることを確認してから
ローカルまたはModal CLIでそのrunの`.running`だけを除去し、resumeしてください。
チェックポイントがない失敗runは別名でやり直します。自動的に既存ディレクトリを消しません。
同一Volume/runへの並行起動はしないでください。複数の独立deploymentをまたぐ厳密な分散lockではありません。

CPUでの同一実行条件では再開一致をテスト済みです。GPU・backend・ライブラリ・デバイス変更後の
bitwise一致は保証しません。CheckpointはPyTorch `weights_only=True`で読むものの、
未知の第三者checkpointを安全と保証するものではありません。

## 学習率・目的の切替

warmup→一定LRで中断・再開します。学習を継続するだけならoptimizerをresetしません。
公開候補のcooldownを開始するときは、元の設定でresumeに`--start-cooldown`を追加します。
この操作はoptimizerと累積カウンタを維持し、現在のtoken数をcooldown開始点として記録します。
次のresumeでは更新された`runs/名前/config.json`を指定してください。warmup前のcooldownは拒否します。
初版のModal wrapperにはこのフラグを公開していません。必要時はrunを取得してローカルCLIで使うか、
同じ検証済みtrainer引数をwrapperへ明示的に追加してください。

SFTや訂正への切替は`--init-from 親run`。親重みを引き継ぎ、新しいoptimizerと目的別counterを開始します。
親の評価用論文ID/group/hashは子に継承します。旧知識をまだ持たないモデルを「忘却しない」と評価しません。

## 更新実験

同じ親から2つの子を作り、Replayなしと20%を比較します。Replayなしでは設定のreplay_ratioを0にし、
`--replay`も省きます。双方の`--new-token-limit`をそろえます。
比較するのは、新しい訂正の直接再現、未見条件への転移、更新前に正答した旧問の保持、更新費用です。
言い換えだけを増やして独立事例数を水増ししません。評価者・正解・合否基準は結果を見る前に固定します。
自動evaluateはNLL/BPBのみです。科学QA判定や95%区間を捏造しません。

## 課金上の注意

jobの時間上限は更新境界でのsoft stopで、保存時間等のreserveがあります。
1更新が長ければreserveを超える可能性があります。Modalのtimeoutは別のhard stopです。
image構築・storage・provider価格変更・他のjobはこの費用推定では制御できません。
Workspace上限と各ジョブの実行量制限を併用し、最初から長いGPU jobを起動しないでください。

## 既知の制限

- Stdlib byte-BPEは正しさを優先したCPU参照実装。大コーパスで速度最適化の対象。
- prepareはbounded JSONL単位。フルarXivの分散ETLやincremental shard結合は未実装。
- SFTはcontext内の1問1答。章分割による超長文生成は未実装。
- 近重複・意味的汚染・引用実在性・科学的検証は人手の工程を残す。
- BF16/SDPAの実GPUbackend、Modal SDK/image、HF importerはこの環境で実行未確認。
- H-Net、ES、GRPO、LoRA、分散学習、API servingは今回の範囲外。
