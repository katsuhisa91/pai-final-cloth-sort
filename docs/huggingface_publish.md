# Hugging Face のモデル・データセットを公開する手順

学習済みの ACT とデモのデータセットは Hugging Face Hub に private で置いてある。公開すると、他の人が
再学習(1 時間半〜2 時間)なしで `--policy katsuhisa91/pai-cloth-sort-act` のように試せる。
コードの再現や課題の採点には必須ではないので、公開は提出後でもよい。

| 種類 | リポジトリ | 中身 |
|---|---|---|
| モデル | [katsuhisa91/pai-cloth-sort-act](https://huggingface.co/katsuhisa91/pai-cloth-sort-act) | ACT v1(20,000 ステップ)。config.json + safetensors + 前処理の定義 |
| データセット | [katsuhisa91/cloth_pickplace](https://huggingface.co/datasets/katsuhisa91/cloth_pickplace) | デモ 123 本(LeRobot v3) |
| データセット | [katsuhisa91/cloth_pickplace_v3](https://huggingface.co/datasets/katsuhisa91/cloth_pickplace_v3) | デモ 452 本(DART 込み、相対状態 12 次元) |

## 1. 公開前の確認

- 中身はシミュレータの画像・状態・行動だけで、実写真や個人情報は含まれていない(MuJoCo の描画のみ)
- 各リポジトリのカード(Hub 上の README.md)に、氏名・所属・メールアドレスが書かれていないこと
- カードのライセンスが `apache-2.0` になっていること(このリポジトリと同じ)。無ければカードの先頭に足す

  ```yaml
  ---
  license: apache-2.0
  ---
  ```

## 2. 公開する

### ブラウザで行う場合

それぞれのリポジトリのページ → Settings → 「Change Repository Visibility」→ Make public。3 つとも行う。

### コマンドで行う場合

`hf auth login` 済み(書き込み権限のあるトークン)の `.venv-il` で:

```sh
.venv-il/bin/python - <<'PY'
from huggingface_hub import HfApi

api = HfApi()
api.update_repo_settings("katsuhisa91/pai-cloth-sort-act", private=False)
for repo in ["katsuhisa91/cloth_pickplace", "katsuhisa91/cloth_pickplace_v3"]:
    api.update_repo_settings(repo, private=False, repo_type="dataset")
PY
```

古い huggingface_hub で `update_repo_settings` が無いと言われたら、`api.update_repo_visibility(repo, private=False, repo_type=...)` に読み替える。

## 3. 公開できたか確かめる

ログインしていない状態で API が 200 を返せば公開されている(private の間は 401 か 404)。

```sh
for u in models/katsuhisa91/pai-cloth-sort-act datasets/katsuhisa91/cloth_pickplace datasets/katsuhisa91/cloth_pickplace_v3; do
  printf '%s ' "$u"; curl -s -o /dev/null -w '%{http_code}\n' "https://huggingface.co/api/$u"
done
```

ブラウザのシークレットウィンドウで 3 つのページが開けるかでも確かめられる。
LeRobot から読めるかは、トークンなしで次が通るかで確かめる。

```sh
HF_TOKEN= .venv-il/bin/python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; print(LeRobotDataset('katsuhisa91/cloth_pickplace').num_episodes)"
```

## 4. 公開した後

- README の「Hugging Face に上げる場合」節の「(2026-10-17 時点で private)」と、「private の間はアクセス権のあるトークンでログインしておく」の注記を消す
- 非公開に戻すのは同じ手順で `private=True`(ブラウザなら Make private)。ただし公開中にダウンロードされたものは取り消せない
