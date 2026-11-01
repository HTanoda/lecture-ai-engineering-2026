# 第2回 基礎課題の実行記録

ユーザーの指示により、Colab T4 の代わりにローカルの NVIDIA GeForce RTX 5070 Ti を使用する。
講座指定のモデル revision、学習データ、評価データ、採点関数、LoRA 設定は配布ノートブックを使用する。

## 比較条件

- 標準: `num_train_epochs=20`（40ステップ）
- 変更: `num_train_epochs=10`（20ステップ）
- 両条件とも Python を終了してから、新しいプロセスでモデルを読み直す。
- 各条件の保存先を分け、既存の実行結果を上書きしない。
- ライブラリはノートブックの PINS を維持する。PyTorch は CUDA 13.0 版の 2.11.0。
- Python はこのPCの 3.12.8。Colab の 3.13 / T4 との差異は提出文に明記する。
- 量子化なし。精度の設定を独自には変更せず、配布コードとライブラリの既定値を使用する。
- 実際の保存済み学習引数を確認したところ、両条件とも `fp16=False`、`bf16=True`、`gradient_checkpointing=True`。基盤モデルをfloat32で読み込み、学習計算にはbf16混合精度が使われた。配布コードの `USE_AMP=False` はfp16にだけ渡され、bf16はTRLの既定値が有効になる。
- 実行スクリプトは両条件で同じ seed 42 をモデル読み込み前にも設定する。

## 再実行

このフォルダーを作業ディレクトリとし、専用仮想環境を用意した後で実行する。
既存結果がある場合は上書きを拒否するため、新しいコピーを作って実行する。

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv\Scripts\python.exe -r requirements.lock.txt
.venv\Scripts\python.exe -u run_local.py --epochs 20
.venv\Scripts\python.exe -u run_local.py --epochs 10
.venv\Scripts\python.exe verify_results.py
```

モデルは初回に公開配布元から取得する。保存先は `.hf-cache/`。
ライブラリとモデルの取得にネットワークを使い、推論・学習には外部APIを使わない。
個人情報を含まない、配布された架空FAQのみを使用する。

## 保存する記録

各 `runs/epochs_20/` と `runs/epochs_10/` に次を保存する。

- `result.json`: 配布ノートブックの結果に、実測学習時間・loss・実行環境を追加
- `eval_before.json` / `eval_after.json`: 21問の回答原文と採点
- `train.jsonl` / `test.jsonl` / `gold.json`: 学習・評価データと正解
- `trainer_state.json`: 学習ステップとlossログ
- `adapter/` / `ckpt/`: 学習モデルとチェックポイント（提出用には含めない）

`final_logged_loss` は最後の10ステップ分のログの `loss`。
全学習期間の平均である `mean_training_loss` とは分けて記録する。
学習時間は CUDA の同期を取り、`trainer.train()` の開始から終了までを測定する。
モデル取得・学習前後の評価・学習後のadapter保存時間は含まない。
チェックポイント保存は学習処理内なので含む。

`executed_training_myself=true` は講師の参考結果の転記ではなく、
このPCで実際に学習を実行したことを表す。操作はユーザーの依頼に基づき Codex が行う。

## 提出

提出先は自分の fork の `lectures/02/answer.md`。
作業ブランチ `lec02` から自分の fork の `main` へ PR を作成する。
コミット・push はユーザーの明示指示後に行う。

## 一次資料

- [配布ノートブック](https://github.com/matsuolab/lecture-ai-engineering-2026/blob/main/notebooks/lora_format_tuning.ipynb)
- [演習・宿題の要領](https://github.com/matsuolab/lecture-ai-engineering-2026/blob/main/lectures/02/problem.md)
- ローカルの「第二回宿題資料.pdf」「第二回演習講義資料.pdf」

