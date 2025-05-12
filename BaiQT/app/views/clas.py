import streamlit as st
from controllers.classification import TextClassifier
import pandas as pd
import os
import matplotlib.pyplot as plt

def classification_view():
    st.title("Phân loại dữ liệu")  

    # Lựa chọn nguồn dữ liệu
    data_source = st.radio("Chọn nguồn dữ liệu", ("Dataset có sẵn", "Tải lên file dữ liệu"))

    dataset_name = None
    uploaded_file = None
    
    if data_source == "Dataset có sẵn":
        dataset_list = [
            "IMDb Review", "Yelp Review", "Amazon Review", 
            "TREC", "Yahoo! Answer", "AG's News", "Sogou News", "DBPedia"
        ]
        dataset_name = st.selectbox("Chọn Dataset", dataset_list)
    else:
        uploaded_file = st.file_uploader("Tải lên file dữ liệu (CSV hoặc TXT)", type=["csv", "txt"])

    # Chọn thuật toán phân loại
    model_list = ["Naive Bayes", "Logistic Regression", "SVM", "K-Nearest Neighbors", "Decision Tree"]
    model_type = st.selectbox("Chọn Thuật toán", model_list)

    # Khởi tạo từ trước nếu chưa có trong session_state
    if "accuracy_dict" not in st.session_state:
        st.session_state["accuracy_dict"] = {}

    if st.button("Train"):
        if data_source == "Tải lên file dữ liệu" and uploaded_file is None:
            st.warning("Vui lòng tải lên file dữ liệu để tiếp tục.")
            return

        classifier = TextClassifier(dataset_name, model_type, uploaded_file)  
        progress_bar = st.progress(0)  
        
        # Huấn luyện mô hình và lấy kết quả
        accuracy, train_time = classifier.train_model(progress_bar)  
        st.success(f"Độ chính xác: {accuracy * 100:.2f}%")

        # Lưu độ chính xác và tên dataset vào dictionary trong session_state
        st.session_state["accuracy_dict"][(dataset_name, model_type)] = accuracy * 100

        st.session_state["classifier"] = classifier

    # Nút để vẽ lại đồ thị
    if st.button("Vẽ lại biểu đồ độ chính xác"):
        if st.session_state["accuracy_dict"]:
            st.markdown("### Đồ thị độ chính xác của các mô hình đã huấn luyện")
            fig, ax = plt.subplots()
            
            dataset_model_pairs = list(st.session_state["accuracy_dict"].keys())
            dataset_names = [pair[0] for pair in dataset_model_pairs]
            model_types = [pair[1] for pair in dataset_model_pairs]
            accuracies = list(st.session_state["accuracy_dict"].values())
            
            ax.barh(range(len(dataset_names)), accuracies, color='skyblue')

            ax.set_yticks(range(len(dataset_names)))
            ax.set_yticklabels([f"{dataset_names[i]} - {model_types[i]}" for i in range(len(dataset_names))])
            
            ax.set_xlabel('Độ chính xác (%)')
            ax.set_title('Độ chính xác của các mô hình phân loại')
            st.pyplot(fig)
        else:
            st.warning("Chưa có dữ liệu để vẽ biểu đồ. Hãy huấn luyện ít nhất một mô hình.")

    # Nút reset biểu đồ
    if st.button("Reset biểu đồ"):
        st.session_state["accuracy_dict"] = {}
        st.success("Đã reset biểu đồ!")

    # Phần dự đoán văn bản mới
    st.markdown("### Dự đoán")
    user_input = st.text_area("Nhập văn bản muốn phân loại", "")

    if st.button("Dự đoán"):
        if "classifier" in st.session_state:
            classifier = st.session_state["classifier"]
            prediction = classifier.predict(user_input)  
            st.info(f"**Dự đoán:** {prediction}") 
        else:
            st.warning("Train mô hình trước khi dự đoán!") 
