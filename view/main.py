import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
print("sys.path:", sys.path)  

from controller.nlpcontroller import NLPController

st.title("Xử lý Ngôn ngữ Tự nhiên")
text = st.text_area("Nhập văn bản của bạn:")

if text:
    #Thu thâp dữ liệu


    #Tăng cường dữ liệu
    st.subheader("Tăng cường dữ liệu")
    aug_types = st.multiselect("Chọn phương thức", ["synonym", "swap", "delete", "noise"])
    if st.button("Tăng cường"):
        st.write(NLPController.augment_text(text, aug_types))
    

    #Xử lý dữ liệu
    st.subheader("Xử lý dữ liệu")
    options = {
        "Tách câu": st.checkbox("Tách câu"),
        "Xóa từ dừng": st.checkbox("Xóa từ dừng"),
        "Xóa dấu câu": st.checkbox("Xóa dấu câu"),
        "Stemming - Lemmatization": st.checkbox("Stemming - Lemmatization"),
        "Nhận diện thực thể (NER)": st.checkbox("Nhận diện thực thể (NER)"),
        "Sửa lỗi chính tả": st.checkbox("Sửa lỗi chính tả"),
        "Sửa viết tắt": st.checkbox("Sửa viết tắt")
    }
    if st.button("Xử lý văn bản"):
        st.write(NLPController.process_text(text, options))


    #Biểu diễn văn bản
    st.subheader("Biểu diễn văn bản")
    method = st.selectbox("Chọn phương pháp", ["One-Hot Encoding", "Bag of Words", "TF-IDF", "GloVe", "Bag of N-Grams"])
    if st.button("Thực hiện mã hóa"):
        st.write(NLPController.represent_text(text, method))
    
    
    
    