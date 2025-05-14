# chatbot.py

from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

# Hàm load chatbot và huấn luyện từ file tiếng Việt
def load_chatbot():
    chatbot = ChatBot(
        "Trợ lý tiếng Việt",
        storage_adapter="chatterbot.storage.SQLStorageAdapter",
        logic_adapters=["chatterbot.logic.BestMatch"],
        database_uri="sqlite:///vietnamese_chatbot.sqlite3"
    )
    
    trainer = ChatterBotCorpusTrainer(chatbot)

    # Đường dẫn tuyệt đối đến file hội thoại tiếng Việt
    data_path = r"D:\HK2_2025\XLNNTN\ProjectXLNNTN\ChatBot\data\hoi_dap.yml"
    trainer.train(data_path)

    return chatbot
