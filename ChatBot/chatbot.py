import openai
import random
import json
import os

import openai

# Cấu hình OpenAI API key
openai.api_key = 'AIzaSyBgYCTqH4FoTUzaMLmyXX5YAMEADB3T27k'

def chatbot_gemini(user_input):
    """Sử dụng OpenAI API để xử lý yêu cầu người dùng"""
    try:
        response = openai.chat.Completion.create(
            model="gpt-3.5-turbo",  # Hoặc "gpt-4" nếu bạn có quyền truy cập
            messages=[{"role": "user", "content": user_input}]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Đã có lỗi xảy ra: {e}"

# Giả sử bạn vẫn muốn sử dụng mã tương tự cho chatbot_local
def chatbot_local(user_input):
    """Chế độ tự train mô hình - sử dụng mô hình đã huấn luyện"""
    # Đọc file intents.json chứa các câu hỏi và câu trả lời của chatbot
    with open("intents.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Giả sử bạn có một mô hình đã huấn luyện hoặc một hàm xử lý đơn giản
    # Tìm câu hỏi khớp nhất với input của người dùng
    tag = "default"  # Giả sử bạn có một cách phân loại câu hỏi
    
    for intent in data['intents']:
        if user_input.lower() in intent['patterns']:
            tag = intent['tag']
            break
    
    # Tìm câu trả lời phù hợp với tag đã chọn
    for intent in data['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    
    return "Xin lỗi, tôi không hiểu yêu cầu của bạn."
