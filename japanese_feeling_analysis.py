import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline, AutoTokenizer

# --------------------------------
# 設定とモデルの読み込み
# --------------------------------

@st.cache_resource
def load_model():
    """【完成版】ライブチャット特化・9感情分析モデルをロードして返す"""
    tokenizer = AutoTokenizer.from_pretrained("tohoku-nlp/bert-base-japanese-v3")
    model = pipeline(
        "text-classification",
        model="iton/YTLive-JaBERT-Emotion-v1",
        tokenizer=tokenizer
    )
    return model

# モデルをロード
try:
    classifier = load_model()
except Exception as e:
    st.error(f"モデルの読み込み中にエラーが発生しました:{e}")
    st.stop()

# --------------------------------
# Streamlit UI部分
# --------------------------------
st.title("テキストからわかる感情分析アプリ〜9種類の感情によるアプローチ〜")
st.write("テキストを入力すると、文章に含まれる9つの感情のバランスを分析します。")

# --- ★★★ 1. AIの内部辞書をDataFrame化して表示 ★★★ ---
# モデルが持つ「ID⇔ラベル」の対応表を取得
id2label = classifier.model.config.id2label
# DataFrameに変換
id2label_df = pd.DataFrame(id2label.items(), columns=['内部ID (番号)', 'ラベル (日本語)'])
id2label_df['内部ID (ラベル名)'] = id2label_df['内部ID (番号)'].apply(lambda x: f"LABEL_{x}")

with st.expander("モデルの内部辞書（id2label）を見てみる"):
    st.write("このモデルは、テキストから感情を予測し、IDと日本語の対応表を使って、分析結果を日本語で表している")
    st.table(id2label_df[['内部ID (ラベル名)', '内部ラベル (日本語)']])
# ----------------------------------------------------

EMOTION_LABELS = {
    'joy': '喜び 😊', 'sadness': '悲しみ 😢', 'anticipate': '期待 ✨', 
    'surprise': '驚き 😮', 'anger': '怒り 😠', 'fear': '恐れ 😨', 
    'disgust': '嫌悪 🤢', 'trust': '信頼 🤗', 'neutral': '中立 😐'
}

user_input = st.text_area(
    "分析したい文章を入力してください",
    "今日はカレーライスだ！うれしい！"
)

if st.button("感情を分析する"):
    if user_input:
        with st.spinner("分析中です..."):
            result = classifier(user_input, return_all_scores=True)
        
        emotions = result[0]
        df = pd.DataFrame(emotions)
        
        # --- ★★★ 2. 分析結果のDataFrameを強化 ★★★ ---
        # 元のラベル列を、分かりやすい名前に変更
        df.rename(columns={'label': 'internal_id'}, inplace=True)
        
        # 内部IDを、内部ラベル（日本語）に翻訳
        df['internal_label_en'] = df['internal_id'].apply(
            lambda x: id2label[int(x.split('_')[1])] if '_' in x else x
        )
        
        # # ラベルから表示用の日本語に変換
        # df['emotion_jp'] = df['internal_label_en'].map(EMOTION_LABELS)
        # ---------------------------------------------

        df_sorted = df.sort_values(by='score', ascending=False)

        st.subheader("分析結果")
        top_emotion = df_sorted.iloc[0]
        st.info(f"主な感情は「{top_emotion['internal_label_en']}」です。 (確信度: {top_emotion['score']:.2%})")

        # --- ★★★ 3. Top3表示を改善 ★★★ ---
        st.subheader("感情スコア Top 3")
        st.write("AIの内部的な判断（ID）と、日本語ラベルを並べて表示します。")
        st.table(
            df_sorted[['internal_label_en', 'score']].head(3).style.format({'score': '{:.2%}'})
        )
        # ----------------------------------

        st.subheader("全感情のバランス（レーダーチャート）")
        fig = px.line_polar(
            df, 
            r='score',
            theta='internal_label_en',
            line_close=True,
            range_r=[0, 1],
            title="感情の構成"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("分析する文章を入力してください。")

st.markdown("---")
st.markdown("このアプリは、Hugging Faceで公開されている事前学習済みモデル [iton/YTLive-JaBERT-Emotion-v1](https://huggingface.co/iton/YTLive-JaBERT-Emotion-v1) を利用しています。")
