import streamlit as st
from controller.classification import TextClassifier

def classification_view():
    st.title("Phân loại dữ liệu")  

    # Chọn dataset để huấn luyện
    dataset_list = [
        "IMDb Review", "Yelp Review", "Amazon Review", 
        "TREC", "Yahoo! Answer", "AG's News", "Sogou News", "DBPedia"
    ]
    dataset_name = st.selectbox("Chọn Dataset", dataset_list)

    # Chọn thuật toán phân loại
    model_list = ["Naive Bayes", "Logistic Regression", "SVM", "K-Nearest Neighbors", "Decision Tree"]
    model_type = st.selectbox("Chọn Thuật toán", model_list)

    if st.button("Train"):
        classifier = TextClassifier(dataset_name, model_type)  

        progress_bar = st.progress(0)  
    
        accuracy, train_time = classifier.train_model(progress_bar)  
        st.success(f"Mô hình '{model_type}' đạt độ chính xác: {accuracy * 100:.4f}%")
        st.write(f"Thời gian huấn luyện: {train_time:.4f} giây")

        st.session_state["classifier"] = classifier

    # Phần dự đoán văn bản mới
    st.markdown("### Dự đoán văn bản mới")
    user_input = st.text_area("Nhập văn bản cần phân loại", "")

    if st.button("Dự đoán"):
        if "classifier" in st.session_state:
            classifier = st.session_state["classifier"]
            prediction = classifier.predict(user_input)  
            st.info(f"**Dự đoán:** {prediction}") 
        else:
            st.warning("Train mô hình trước khi dự đoán!")  
