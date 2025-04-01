import streamlit as st
import pandas as pd
from controller.representation import DataRepresentation

def representation_view():
    st.title("Biểu Diễn Dữ Liệu")

    # Chọn cách nhập dữ liệu
    input_type = st.radio("Nguồn dữ liệu:", ("Nhập dữ liệu", "Tải file (.txt)"))

    text_data = []
    if input_type == "Nhập dữ liệu":
        input_text = st.text_area("Nhập dữ liệu:", height=200)
        text_data = input_text.split("\n") if input_text else []
    elif input_type == "Tải file (.txt)":
        uploaded_file = st.file_uploader("Chọn file .txt", type="txt", key="file_upload_key")
        if uploaded_file:
            text_data = uploaded_file.read().decode("utf-8").split("\n")

    # Chọn phương pháp biểu diễn
    method = st.radio("Chọn phương pháp biểu diễn:", [
        "One-hot Encoding", "CountVectorizer", "Bag of N-grams",
        "TF-IDF Vectorizer", "Word2Vec Embedding", "GloVe Embedding",
        "FastText Embedding", "ChatGPT Embedding"
    ])

    if st.button("Biểu diễn dữ liệu"):
        if text_data:
            method_map = {
                "One-hot Encoding": "onehot",
                "CountVectorizer": "count",
                "Bag of N-grams": "bagofngram",
                "TF-IDF Vectorizer": "tfidf",
                "Word2Vec Embedding": "word2vec",
                "GloVe Embedding": "glove",
                "FastText Embedding": "fasttext",
                "ChatGPT Embedding": "chatgpt",
                "BERT Embedding": "bert",
            }
            vectorizer = DataRepresentation(method=method_map.get(method, "count"))
            result = vectorizer.fit_transform(text_data)

            if not isinstance(result, pd.DataFrame):
                result = pd.DataFrame(result)

            if isinstance(result, pd.DataFrame):
                st.subheader("Kết quả Biểu Diễn Dữ Liệu")
                st.dataframe(result)

                csv_data = result.to_csv(index=False).encode("utf-8")
                st.download_button("Tải kết quả CSV", data=csv_data, file_name="vectorized_data.csv", mime="text/csv")
            else:
                st.warning(result)
        else:
            st.warning("Vui lòng nhập dữ liệu!")
