## 日本語 9感情分析アプリ（README）

このリポジトリは、Hugging Face の事前学習モデルを活用して、日本語テキストを 9種類の感情に分類・可視化するアプリケーションのコードと解説をまとめたものです。

Streamlit を用いた Web アプリとして、感情スコアの一覧表示やレーダーチャートによる可視化を行います。

## 特長

- **日本語特化モデルを使用**
    - Tokenizer: `tohoku-nlp/bert-base-japanese-v3`
    - 感情分析モデル: `iton/YTLive-JaBERT-Emotion-v1`
- **9感情分類**
    - joy / sadness / anticipate / surprise / anger / fear / disgust / trust / neutral
- **内部プロセスの可視化**
    - モデル内部の `id2label` を DataFrame として表示
    - 「内部ID」「英語ラベル」といった内部表現を確認可能
- **インタラクティブな可視化**
    - Top3 感情のスコア表示
    - Plotly によるレーダーチャートで全感情のバランスを表示

## セットアップ

### 1. 必要環境

- Python 3.9 以降（推奨）
- インターネット接続（Hugging Face からモデルを取得）

### 2. 必要ライブラリのインストール

```bash
pip install streamlit transformers pandas plotly
pip install fugashi ipadic
```

※ 日本語モデル利用のため、`fugashi` や `ipadic` などの依存ライブラリが必要です。

## 使い方

### 1. アプリの起動

```bash
streamlit run [app.py](http://app.py)
```

※ 本README内のコードを [`app.py`](http://app.py) として保存した想定です。

### 2. 画面の操作

1. 画面上部にアプリタイトルと説明が表示されます。
2. テキストエリアに、感情分析したい日本語テキストを入力します。
3. 「感情を分析する」ボタンをクリックします。
4. 以下の情報が表示されます。
    - 主な感情とその確信度
    - 感情スコア Top3 の一覧
    - 9感情全体のスコアを示すレーダーチャート
    - 必要に応じて、`id2label` の内部辞書テーブル（エクスパンダ内）

## コード構成（概要）

### モデルロード部分

```python
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("tohoku-nlp/bert-base-japanese-v3")
    model = pipeline(
        "text-classification",
        model="iton/YTLive-JaBERT-Emotion-v1",
        tokenizer=tokenizer
    )
    return model
```

- 初回アクセス時のみモデルをロードし、以降はキャッシュを利用
- モデル読み込み失敗時はエラーメッセージを表示し処理を停止

### 内部辞書（id2label）の可視化

```python
id2label = [classifier.model.config.id](http://classifier.model.config.id)2label
id2label_df = pd.DataFrame(id2label.items(), columns=['内部ID (番号)', '内部ラベル (英語)'])
id2label_df['内部ID (ラベル名)'] = id2label_df['内部ID (番号)'].apply(lambda x: f"LABEL_{x}")
```

- モデル内部のクラスIDとラベル名をテーブル化
- ユーザーがモデルの「認識している感情ラベル」を確認できる

### 推論と結果表示

- `classifier(user_input, return_all_scores=True)` で9感情すべてのスコアを取得
- DataFrame に変換し、スコア順にソート
- Top1 を主感情として表示
- Top3 をテーブル表示
- Plotly の `line_polar` を用いたレーダーチャートで可視化

## 想定ユースケース

- **顧客サポート・VoC分析**
    - 怒り・嫌悪の高い問い合わせの優先度付け
    - 期待スコアの高いフィードバックから改善ポイントを抽出
- **SNS・ライブ配信の分析**
    - コメントの感情推移を可視化し、炎上や離脱の兆候を早期検知
- **マーケティング**
    - 広告コピーやキャンペーン文言の印象を事前に定量チェック
- **組織開発・人事**
    - 匿名アンケートの自由記述を感情分析し、組織状態を把握

## 今後の拡張アイデア

- 会話単位での感情推移の時系列分析
- 感情変化のトリガーとなる文やキーワードの自動抽出
- LLM と連携した「なぜその感情になったか」の説明生成
- ダッシュボード化による継続的モニタリング

## 参考

- 使用モデル:
    - Hugging Face: `iton/YTLive-JaBERT-Emotion-v1`
    - Tokenizer: `tohoku-nlp/bert-base-japanese-v3`
