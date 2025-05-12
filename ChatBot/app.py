import streamlit as st
from chatbot import chatbot_gemini, chatbot_local

st.title("🤖 Chatbot với Streamlit")

# Chọn chế độ sử dụng API hoặc tự train mô hình
mode = st.sidebar.radio("Chọn chế độ", ["Dùng Google Cloud API", "Tự train mô hình"])

user_input = st.text_input("Bạn:", "")

# Xử lý khi có input từ người dùng
if user_input:
    if mode == "Dùng Google Cloud API":
        response = chatbot_gemini(user_input)
        st.text_area("Bot:", value=response, height=100)
    else:
        response = chatbot_local(user_input)
        st.text_area("Bot:", value=response, height=100)
