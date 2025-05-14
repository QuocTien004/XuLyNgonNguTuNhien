# chatbot_app.py

import streamlit as st
from chatbot import load_chatbot  # Import hàm load_chatbot từ chatbot.py

# Cấu hình giao diện Streamlit
st.set_page_config(page_title="Chatbot tiếng Việt", page_icon="💬")
st.title("💬 Chatbot")

# Khởi tạo chatbot
chatbot = load_chatbot()

# Giao tiếp với người dùng
user_input = st.text_input("👤 Bạn:", "")

if user_input:
    response = chatbot.get_response(user_input)
    st.markdown(f"**Bot**: {response}")
