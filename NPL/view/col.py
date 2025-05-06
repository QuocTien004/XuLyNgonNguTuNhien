from controller.collection import DataDownloader
import streamlit as st
import pandas as pd

def scraper_view():
    st.title("Lấy dữ liệu từ trang web") 

    if "scraper" not in st.session_state:
        st.session_state.scraper = DataDownloader()
    if "sub_classes" not in st.session_state:
        st.session_state.sub_classes = []
    if "all_classes" not in st.session_state:
        st.session_state.all_classes = []

    # Nhập URL
    url = st.text_input("Nhập URL của trang web:")

    if url and st.button("Lấy dữ liệu từ web"):
        html_preview = st.session_state.scraper.fetch_webpage(url)  

        with st.expander("Xem chi tiết"):
            st.code(html_preview, language="html")  

        st.session_state.all_classes = st.session_state.scraper.get_all_classes()
