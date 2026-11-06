# データ契約

## 原論文JSONL（1行1論文）

```json
{"paper_id":"実際の論文ID","group_id":"同一研究・版をまとめるID","version":"確認した版","license":"https://creativecommons.org/licenses/by/4.0/","kind":"paper","text":"本文とLaTeX"}
```

これは形式説明であり学習データではありません。版が不明なassembled TeXは`version:null`として扱い、
初投稿日だけで「訂正前の知識」と推定しません。ソース改訂の紐付けは別の監査対象です。

入力は最大256MiB（変更可能）でメモリ内のID・重複groupを確定してから書き出します。
最初の1000文字未満、NULを含む資料、短いPDF wrapper等を除外します。
安全のためTeX実行・無制限macro展開・スクリプト実行はありません。
完全な品質フィルタや近重複判定ではありません。

同一paper IDの版違い、明示的group、本文の完全重複をunionしてから
決定的hashで90/5/5相当のtrain/validation/testへ割り当てます。
小標本で割合や非空性は保証しません。Tokenizerにはtrain資料だけを渡します。
追加コーパスは既存Tokenizerを`--tokenizer`で渡します。

## 論文由来SFT・検証済み訂正

```json
{"paper_id":"訂正論文ID","group_id":"研究単位","version":"v2","license":"https://creativecommons.org/licenses/by/4.0/","kind":"correction","prompt":"科学的質問","answer":"検証された短い論文形式の説明","verified":true,"verification":"検証者・検証手順・結果の記録","evidence":[{"paper_id":"根拠論文ID","version":"v2","span":"対象section・式番号等"}]}
```

`kind`は`paper`、`sft`、`correction`。SFTとcorrectionは検証記録・evidenceが必要です。
`verified:true`は利用者のattestationであって、システムが科学的に検証したという意味ではありません。
誤った主張を正解文として流し込まず、誤り・修正・適用条件・未解決点を含む教師を用意します。

教師データはBOS + prompt + SEP + answer + EOSとして符号化します。
prompt/BOS/SEPの位置は損失対象外、answer/EOSだけを予測対象にします。
SFTの全例がcontextに入らない場合は拒否します。質問を切り捨てたまま答えだけを学習しません。
原論文は文書内の連続窓として学習し、窓・batchが別論文をまたぐことはありません。

## 生成されるローカル成果物

- tokenizer.json: 可逆byte-BPE、merge順序、語彙容量。学習後は不変。
- tokens.bin / mask.bin / bytes.bin: uint16 token、uint8 target mask、uint16 token byte length。
- records.json: 論文ID・group・split・provenance・各配列のstart/end。
- manifest.json: 全ファイルのSHA-256、入力hash、Tokenizer identity、採否件数。

ハッシュは整合性検証であり、悪意のある書換えに対する署名ではありません。
信頼できる自分の成果物だけを読み込んでください。
元のコーパス本文はGitHubに置きません。科学的品質・ライセンスは人手監査が必要です。
