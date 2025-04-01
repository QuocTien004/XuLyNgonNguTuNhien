import streamlit as st
import nltk
from view.aug import augmentation_view
from view.col import scraper_view
from view.pre import preprocessor_view
from view.rep import representation_view
from view.clas import classification_view

nltk.download('averaged_perceptron_tagger_eng')

def main():
    st.set_page_config(page_title="Xử lý Dữ Liệu NLP", layout="wide")

    st.sidebar.title("Menu Chức Năng")
    choice = st.sidebar.radio("Chọn một tác vụ:", 
                              ["Thu thập dữ liệu", "Tăng cường dữ liệu", 
                               "Tiền xử lý", "Biểu diễn dữ liệu", 
                               "Phân loại văn bản"])

    if choice == "Thu thập dữ liệu":
        scraper_view()
    elif choice == "Tăng cường dữ liệu":
        augmentation_view()
    elif choice == "Tiền xử lý":
        preprocessor_view()
    elif choice == "Biểu diễn dữ liệu":
        representation_view()
    elif choice == "Phân loại văn bản":
        classification_view()

if __name__ == "__main__":
    main()
