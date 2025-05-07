import json
import random
import re

with open('data/intents.json', 'r') as file:
    intents = json.load(file)

def clean_text(text):
    return re.sub(r"[^\w\s]", "", text.lower())

def get_response(user_input):
    user_input = clean_text(user_input)
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            pattern_clean = clean_text(pattern)
            if pattern_clean in user_input:
                return random.choice(intent["responses"])
    # nếu không khớp thì dùng intent "noanswer"
    for intent in intents["intents"]:
        if intent["tag"] == "noanswer":
            return random.choice(intent["responses"])
    return "I'm not sure I understand."
