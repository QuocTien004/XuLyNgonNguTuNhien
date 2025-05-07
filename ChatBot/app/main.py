import streamlit as st
from chat.bot import get_response

st.set_page_config(page_title="Simple Chatbot", layout="centered")
st.title("💬 Simple Chatbot with Streamlit")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:", key="input")

if st.button("Send") and user_input:
    response = get_response(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

for speaker, message in st.session_state.chat_history:
    if speaker == "You":
        st.markdown(f"**🧑 You**: {message}")
    else:
        st.markdown(f"**🤖 Bot**: {message}")
