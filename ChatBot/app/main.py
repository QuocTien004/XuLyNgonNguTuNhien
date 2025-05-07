import openai
import streamlit as st

# Cấu hình API key của OpenAI
openai.api_key = 'sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'  # Thay thế bằng API key của bạn

# Hàm để gọi API của OpenAI
def get_openai_response(user_input):
    response = openai.Completion.create(  # Đúng phương thức
        model="gpt-3.5-turbo",  # Hoặc "gpt-4" nếu bạn có quyền truy cập
        prompt=user_input,      # Chuyển từ messages sang prompt
        max_tokens=100,         # Giới hạn số lượng token của phản hồi
        temperature=0.7,        # Độ sáng tạo của câu trả lời
        n=1,                    # Số lượng phản hồi bạn muốn nhận
        stop=None               # Kết thúc phản hồi nếu gặp ký tự nhất định
    )
    return response.choices[0].text.strip()

# Giao diện Streamlit
st.title("Chatbot với Streamlit và OpenAI")
st.write("Hãy gõ câu hỏi hoặc nhắn tin cho chatbot!")

# Giao diện nhập liệu của người dùng
user_input = st.text_input("Nhập câu hỏi:")

# Xử lý khi người dùng nhập
if user_input:
    response = get_openai_response(user_input)
    st.write(f"Chatbot: {response}")
