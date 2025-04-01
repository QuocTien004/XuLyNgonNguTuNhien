import streamlit as st
from controller.preprocessing import TextPreprocessor  
import streamlit.components.v1 as components  

def preprocessor_view():
    st.title("Tiền xử lý dữ liệu")
    st.write("Nhập dữ liệu hoặc tải lên file")

    # Chọn cách nhập dữ liệu
    option = st.radio("Nguồn nhập dữ liệu:", ("Nhập dữ liệu", "Tải file"))

    text = ""
    if option == "Nhập dữ liệu":
        text = st.text_area("Nhập dữ liệu của bạn tại đây:", height=200)

    elif option == "Tải file":
        uploaded_file = st.file_uploader("Chọn file .txt", type="txt")
        if uploaded_file:
            text = uploaded_file.read().decode("utf-8")

    if text:
        preprocessor = TextPreprocessor(text)

        # Chọn các phương thức tiền xử lý
        st.subheader("Chọn các phương thức tiền xử lý")
        selected_methods = st.multiselect(
            "Chọn các tùy chọn tiền xử lý:",
            [
                "Tách câu", "Tách từ", "stopwords", "Chuyển thành chữ thường", 
                "Stemming", "Lemmatization", "POS Tagging", 
                "Sửa viết tắt", "Sửa chính tả", "NER"
            ]
        )

        if st.button("Thực hiện", key="run_button"):
            if text.strip():
                st.subheader("Kết quả Xử Lý")
                processed_text = "**Kết quả tiền xử lý:**\n\n"

                methods = {
                    "Tách câu": preprocessor.sentence_tokenization,
                    "Tách từ": preprocessor.word_tokenization,
                    "stopwords": preprocessor.remove_stopwords_punctuation,
                    "Chuyển thành chữ thường": preprocessor.to_lowercase,
                    "Stemming": preprocessor.stemming,
                    "Lemmatization": preprocessor.lemmatization,
                    "POS Tagging": preprocessor.pos_tagging,
                    "Sửa viết tắt": preprocessor.expand_contractions,
                    "Sửa chính tả": preprocessor.correct_spelling,
                    "NER": preprocessor.named_entity_recognition,
                }

                for method, func in methods.items():
                    if method in selected_methods:
                        result = func()
                        st.write(f"**{method}:**", result)
                        processed_text += f"**{method}:**\n{result}\n\n"

                st.download_button("Tải xuống kết quả", data=processed_text, file_name="processed_text.txt")
            else:
                st.warning("Vui lòng nhập dữ liệu!")
