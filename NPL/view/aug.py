import streamlit as st
from controller.augmentation import augmenter

def augmentation_view():
    st.title("Tăng cường dữ liệu")  

    # Chọn cách nhập dữ liệu
    option = st.radio("Nguồn nhập dữ liệu:", ("Nhập văn bản", "Tải file văn bản"))

    text = ""

    # Nhập văn bản thủ công
    if option == "Nhập văn bản":
        text = st.text_area("Nhập văn bản của bạn tại đây:", height=200)

    # Tải file văn bản
    elif option == "Tải file văn bản":
        uploaded_file = st.file_uploader("Chọn file .txt", type="txt")
        if uploaded_file:
            text = uploaded_file.read().decode("utf-8")

    # Chọn phương pháp tăng cường
    methods = st.multiselect("Chọn phương pháp tăng cường:", [
        "Thay từ đồng nghĩa",
        "Đảo vị trí từ",
        "Xóa từ",
        "Thêm từ ngẫu nhiên",
        "Dịch ngược"
    ])

    # Khi nhấn nút "Thực hiện"
    if st.button("Thực hiện"):
        if text.strip():
            results = [text]

            # Áp dụng từng phương pháp đã chọn
            for method in methods:
                augmented_texts = []
                for t in results:
                    if method == "Thay từ đồng nghĩa":
                        augmented_texts.append(augmenter.synonym_augmentation(t))
                    elif method == "Đảo vị trí từ":
                        augmented_texts.append(augmenter.swap_words(t))
                    elif method == "Xóa từ":
                        augmented_texts.append(augmenter.delete_words(t))
                    elif method == "Thêm từ ngẫu nhiên":
                        augmented_texts.append(augmenter.insert_words(t))
                    elif method == "Dịch ngược":
                        augmented_texts.append(augmenter.back_translation(t))
                
                results = augmented_texts  

            # Hiển thị và cho phép tải kết quả
            final_result = "\n\n".join(results)
            st.text_area("Kết quả:", value=final_result, height=150)
            st.download_button("Tải xuống", data=final_result, file_name="augmented_text.txt")
        else:
            st.warning("Vui lòng nhập văn bản!") 