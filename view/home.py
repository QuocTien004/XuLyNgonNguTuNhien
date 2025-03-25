import streamlit as st

def Menu():
    st.sidebar.title("Menu")
    menu = [
        "Tăng cường dữ liệu",
        "Thu thập dữ liệu",
        "Tiền xử lý dữ liệu",
        "Biểu diễn dữ liệu",
        "Phân loại dữ liệu"
    ]
    return st.sidebar.selectbox("Chọn chức năng", menu)
