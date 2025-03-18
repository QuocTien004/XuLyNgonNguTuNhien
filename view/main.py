import streamlit as st
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from controller.nlpcontroller import NLPController

# --- THU THẬP DỮ LIỆU ---
st.subheader("1. Thu thập dữ liệu")
data_collection_method = st.radio("Chọn phương thức:", ["Cào dữ liệu", "Nhập dữ liệu", "Tải dữ liệu"])
user_input_data = ""

if data_collection_method == "Cào dữ liệu":
    url = st.text_input("Nhập URL:")
    if st.button("Cào dữ liệu"):
        user_input_data = NLPController.scrape_data(url)
elif data_collection_method == "Nhập dữ liệu":
    user_input_data = st.text_area("Nhập dữ liệu:")
elif data_collection_method == "Tải dữ liệu":
    uploaded_file = st.file_uploader("Tải lên tệp dữ liệu:", type=["txt", "csv", "json"])
    if uploaded_file is not None:
        user_input_data = uploaded_file.getvalue().decode("utf-8")
        st.text_area("Dữ liệu đã tải:", user_input_data, height=200)

# --- TĂNG CƯỜNG DỮ LIỆU ---
st.subheader("2. Tăng cường dữ liệu")
aug_types = st.multiselect("Chọn phương thức tăng cường:", ["synonym", "swap", "delete", "noise"])
if st.button("Tăng cường"):
    st.write(NLPController.augment_text(user_input_data, aug_types))

# --- XỬ LÝ DỮ LIỆU ---
st.subheader("3. Xử lý dữu liệu")
options = {
    "Tách câu": st.checkbox("Tách câu"),
    "Xóa từ dừng": st.checkbox("Xóa từ dừng"),
    "Xóa dấu câu": st.checkbox("Xóa dấu câu"),
    "Stemming - Lemmatization": st.checkbox("Stemming - Lemmatization"),
    "Nhận diện thực thể (NER)": st.checkbox("Nhận diện thực thể (NER)"),
    "Sửa lỗi chính tả": st.checkbox("Sửa lỗi chính tả"),
    "Sửa viết tắt": st.checkbox("Sửa viết tắt")
}
if st.button("Xử lý dữ liệu"):
    st.write(NLPController.process_text(user_input_data, options))

# --- BIỂU DIỄN DỮ LIỆU ---
st.subheader("4. Biểu diễn dữ liệu")
method = st.selectbox("Chọn phương pháp:", ["One-Hot Encoding", "Bag of Words", "Bag of N-Grams", "TF-IDF", "GloVe", "FastText"])
if st.button("Thực hiện mã hóa"):
    if user_input_data.strip():
        result = NLPController.represent_text(user_input_data, method)
        st.write("Kết quả mã hóa:")
        st.dataframe(result)
    else:
        st.warning("Vui lòng nhập văn bản trước khi thực hiện mã hóa.")

# --- PHÂN LOẠI DỮ LIỆU ---
st.subheader("5. Phân loại dữ liệu")
classifier = st.selectbox("Chọn phương pháp phân loại:", ["Naive Bayes", "Maximizing Likelihood", "SVM"])

# Khởi tạo session_state nếu chưa có
if 'train_data' not in st.session_state:
    st.session_state.train_data = []  
if 'labels' not in st.session_state:
    st.session_state.labels = []  
if 'model' not in st.session_state:
    st.session_state.model = None

# --- Nhập dữ liệu ---
st.subheader("Nhập Dữ Liệu Huấn Luyện")
st.write("Nhập các vector")

vector_text = st.text_area("Các phần tử cách nhau bằng dấu phẩy hoặc khoản trắng. Mỗi dòng là 1 vector", "")

if st.button("Lưu dữ liệu"):
    vector_lines = vector_text.strip().split("\n")
    processed_vectors = []

    try:
        for line in vector_lines:
            vector = list(map(float, line.replace(',', ' ').split()))
            processed_vectors.append(vector)

        if processed_vectors:
            st.session_state.current_vectors = processed_vectors
            st.success(f"Đã nhập {len(processed_vectors)} vector. Nhập nhãn để lưu!")
        else:
            st.warning("Vui lòng nhập ít nhất 1 vector.")
    except ValueError:
        st.error("Lỗi! Vui lòng nhập số hợp lệ.")

if 'current_vectors' in st.session_state and st.session_state.current_vectors:
    label = st.text_input("Nhập nhãn cho các vector vừa nhập", "")

    if st.button("Lưu mẫu huấn luyện"):
        if label.strip():
            st.session_state.train_data.append(st.session_state.current_vectors)
            st.session_state.labels.append(label)
            st.session_state.current_vectors = []  
            st.success(f"Đã lưu mẫu với nhãn '{label}'!")
        else:
            st.warning("Vui lòng nhập nhãn trước khi lưu!")

# --- Hiển thị dữ liệu đã lưu ---
st.subheader("Danh sách mẫu huấn luyện")
for idx, (vectors, label) in enumerate(zip(st.session_state.train_data, st.session_state.labels)):
    st.write(f"**Mẫu {idx + 1} (Nhãn: {label}):**")
    for v_idx, vector in enumerate(vectors):
        st.write(f"  - Vector {v_idx + 1}: {vector}")

# --- Huấn luyện mô hình ---
if st.button("Huấn luyện mô hình"):
    if st.session_state.train_data:
        X_train = [vector for sample in st.session_state.train_data for vector in sample]
        y_train = [label for i, label in enumerate(st.session_state.labels) for _ in st.session_state.train_data[i]]

        if classifier == "Naive Bayes":
            st.session_state.model = NLPController.train_naive_bayes(np.array(X_train), np.array(y_train))
            st.success("Mô hình Naive Bayes đã được huấn luyện thành công!")
        elif classifier == "Maximizing Likelihood":
            st.session_state.model = NLPController.train_maximizing_likelihood(np.array(X_train), np.array(y_train))
            st.success("Mô hình Maximizing Likelihood đã được huấn luyện thành công!")
        elif classifier == "SVM":
            st.session_state.model = NLPController.train_SVM(np.array(X_train), np.array(y_train))
            st.success("Mô hình SVM đã được huấn luyện thành công!")
    else:
        st.warning("Vui lòng nhập ít nhất một mẫu huấn luyện!")

# --- Dự đoán ---
if 'model' in st.session_state and st.session_state.model is not None:
    st.subheader("Dự đoán dữ liệu mới")
    input_text = st.text_input("Nhập vector cần dự đoán ", "")

    if st.button("Dự đoán"):
        try:
            input_features = list(map(float, input_text.replace(',', ' ').split()))
            if classifier == "Naive Bayes":
                predicted_class, predicted_proba = NLPController.predict_naive_bayes(st.session_state.model, input_features)
            elif classifier == "Maximizing Likelihood":
                predicted_class, predicted_proba = NLPController.predict_maximizing_likelihood(st.session_state.model, input_features)
            elif classifier == "SVM":
                predicted_class, predicted_proba = NLPController.predict_SVM(st.session_state.model, input_features)

            st.write(f"**Kết quả dự đoán:** {predicted_class}")
            st.write("**Xác suất cho từng lớp:**")
            st.write(predicted_proba)
        except ValueError:
            st.error("Lỗi! Vui lòng nhập số hợp lệ.")