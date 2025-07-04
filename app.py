# import threading
# from flask import Flask, request, jsonify
# from transformers import AutoTokenizer, AutoModelForQuestionAnswering
# from sentence_transformers import SentenceTransformer
# import pandas as pd
# import torch
# from flask_cors import CORS
# import os

# app = Flask(__name__)
# CORS(app)

# # Global variables for model and data
# tokenizer = None
# model = None
# embedding_model = None
# question_embeddings = None
# questions = None
# answers = None
# models_loaded = threading.Event()

# def load_models_if_needed():
#     global tokenizer, model, embedding_model, question_embeddings, questions, answers
#     if not models_loaded.is_set():
#         print("Loading models on first request...")
#         try:
#             print("Attempting to load tokenizer and model from Hugging Face...")
#             tokenizer = AutoTokenizer.from_pretrained("Haseebay/educare-chatbot")
#             print("Tokenizer loaded successfully.")
#             model = AutoModelForQuestionAnswering.from_pretrained("Haseebay/educare-chatbot")
#             print("Model loaded successfully.")
#             print("Loading sentence transformer...")
#             embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
#             print("Sentence transformer loaded successfully.")
#             print("Loading Q&A dataset...")
#             df = pd.read_excel(os.path.join(app.root_path, "autism_faqs.xlsx"))
#             questions = df["Question"].fillna("").tolist()
#             answers = df["Answer"].fillna("").tolist()
#             question_embeddings = embedding_model.encode(questions, convert_to_tensor=True)
#             print("Q&A dataset loaded successfully.")
#             print("Models and data loaded successfully!")
#         except Exception as e:
#             print(f"Error loading models: {str(e)}")
#             raise
#         finally:
#             models_loaded.set()

# @app.route("/health", methods=["GET"])
# def health():
#     # Simplified health check for bots to ping
#     return jsonify({"status": "healthy", "timestamp": str(datetime.datetime.now())}), 200

# @app.route("/chat", methods=["POST"])
# def chat():
#     load_models_if_needed()
#     # Wait for models to be loaded if not already
#     if not models_loaded.is_set():
#         print("Waiting for models to load...")
#         models_loaded.wait()

#     if tokenizer is None or model is None or embedding_model is None:
#         return jsonify({"error": "Models failed to load. Please try again later."}), 500

#     data = request.get_json()
#     user_question = data.get("message", "").lower().strip()

#     thank_keywords = ["thank", "thanks", "thank you", "shukriya", "thnx", "appreciate"]
#     if any(keyword in user_question for keyword in thank_keywords):
#         return jsonify({"reply": "You're welcome! Let me know if you have more questions related to Autism.😊"})

#     how_are_you_keywords = ["how are you", "how r u", "how's it going", "how are u"]
#     if any(keyword in user_question.lower() for keyword in how_are_you_keywords):
#         return jsonify({"reply": "I'm just an Educare bot, but I'm here to help you! 😊 How can I assist you today?"})

#     input_embedding = embedding_model.encode(user_question, convert_to_tensor=True)
#     scores = torch.nn.functional.cosine_similarity(input_embedding, question_embeddings)
#     best_score = torch.max(scores).item()
#     best_index = torch.argmax(scores).item()

#     if best_score < 0.5:
#         return jsonify({"reply": "Sorry, I cannot answer that question. You can ask me any question about Autism."})

#     return jsonify({"reply": answers[best_index]})

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8000, debug=True)


# import threading
# from flask import Flask, request, jsonify
# from transformers import AutoTokenizer, AutoModelForQuestionAnswering
# from sentence_transformers import SentenceTransformer
# import pandas as pd
# import torch
# from flask_cors import CORS
# import os
# import psutil
# import sys

# print(f"Python version: {sys.version}")

# app = Flask(__name__)
# CORS(app)

# # Global variables for model and data
# tokenizer = None
# model = None
# embedding_model = None
# question_embeddings = None
# questions = None
# answers = None
# models_loaded = threading.Event()

# def load_models_if_needed():
#     global tokenizer, model, embedding_model, question_embeddings, questions, answers
#     if not models_loaded.is_set():
#         print("Loading models on first request...")
#         try:
#             print("Attempting to load tokenizer and model from Hugging Face...")
#             tokenizer = AutoTokenizer.from_pretrained("Haseebay/educare-chatbot")
#             print("Tokenizer loaded successfully.")
#             model = AutoModelForQuestionAnswering.from_pretrained("Haseebay/educare-chatbot")
#             # Apply dynamic quantization to reduce memory usage
#             model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
#             print("Model loaded and quantized successfully.")
#             print("Loading sentence transformer...")
#             embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
#             print("Sentence transformer loaded successfully.")
#             print("Loading Q&A dataset...")
#             df = pd.read_excel(os.path.join(app.root_path, "autism_faqs.xlsx"))
#             questions = df["Question"].fillna("").tolist()
#             answers = df["Answer"].fillna("").tolist()
#             question_embeddings = embedding_model.encode(questions, convert_to_tensor=True)
#             print("Q&A dataset loaded successfully.")
#             print("Models and data loaded successfully!")
#             # Log memory usage
#             memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
#             print(f"Memory usage after loading models: {memory_usage:.2f} MB")
#         except Exception as e:
#             print(f"Error loading models: {str(e)}")
#             raise
#         finally:
#             models_loaded.set()

# @app.route("/health", methods=["GET"])
# def health():
#     return jsonify({"status": "healthy", "timestamp": str(pd.Timestamp.now())}), 200

# @app.route("/chat", methods=["POST"])
# def chat():
#     load_models_if_needed()
#     # Wait for models to be loaded if not already
#     if not models_loaded.is_set():
#         print("Waiting for models to load...")
#         models_loaded.wait()

#     if tokenizer is None or model is None or embedding_model is None:
#         return jsonify({"error": "Models failed to load. Please try again later."}), 500

#     data = request.get_json()
#     user_question = data.get("message", "").lower().strip()

#     thank_keywords = ["thank", "thanks", "thank you", "shukriya", "thnx", "appreciate"]
#     if any(keyword in user_question for keyword in thank_keywords):
#         return jsonify({"reply": "You're welcome! Let me know if you have more questions related to Autism.😊"})

#     how_are_you_keywords = ["how are you", "how r u", "how's it going", "how are u"]
#     if any(keyword in user_question.lower() for keyword in how_are_you_keywords):
#         return jsonify({"reply": "I'm just an Educare bot, but I'm here to help you! 😊 How can I assist you today?"})

#     input_embedding = embedding_model.encode(user_question, convert_to_tensor=True)
#     scores = torch.nn.functional.cosine_similarity(input_embedding, question_embeddings)
#     best_score = torch.max(scores).item()
#     best_index = torch.argmax(scores).item()

#     if best_score < 0.5:
#         return jsonify({"reply": "Sorry, I cannot answer that question. You can ask me any question about Autism."})

#     return jsonify({"reply": answers[best_index]})

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8000))
#     app.run(host="0.0.0.0", port=port)


import threading
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer
import pandas as pd
import torch
from flask_cors import CORS
import os
import psutil
import sys

print(f"Python version: {sys.version}")
print(f"Server started on port: {os.environ.get('PORT', '8000')}")

app = Flask(__name__)
CORS(app)

# Global variables for model and data
tokenizer = None
model = None
embedding_model = None
question_embeddings = None
questions = None
answers = None
models_loaded = threading.Event()

def load_models_if_needed():
    global tokenizer, model, embedding_model, question_embeddings, questions, answers
    if not models_loaded.is_set():
        print("Loading models on first request...")
        try:
            print("Attempting to load tokenizer and model from Hugging Face...")
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
            print("Tokenizer loaded successfully.")
            model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
            # Apply dynamic quantization to reduce memory usage
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            print("Model loaded and quantized successfully.")
            print("Loading sentence transformer...")
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("Sentence transformer loaded successfully.")
            print("Loading Q&A dataset...")
            df = pd.read_csv(os.path.join(app.root_path, "autism_faqs.csv"))
            questions = df["Question"].fillna("").tolist()
            answers = df["Answer"].fillna("").tolist()
            question_embeddings = embedding_model.encode(questions, convert_to_tensor=False)
            print("Q&A dataset loaded successfully.")
            print("Models and data loaded successfully!")
            # Log memory usage
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            print(f"Memory usage after loading models: {memory_usage:.2f} MB")
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise
        finally:
            models_loaded.set()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "timestamp": str(pd.Timestamp.now())}), 200

@app.route("/chat", methods=["POST"])
def chat():
    load_models_if_needed()
    # Wait for models to be loaded if not already
    if not models_loaded.is_set():
        print("Waiting for models to load...")
        models_loaded.wait()

    if tokenizer is None or model is None or embedding_model is None:
        return jsonify({"error": "Models failed to load. Please try again later."}), 500

    data = request.get_json()
    user_question = data.get("message", "").lower().strip()

    thank_keywords = ["thank", "thanks", "thank you", "shukriya", "thnx", "appreciate"]
    if any(keyword in user_question for keyword in thank_keywords):
        return jsonify({"reply": "You're welcome! Let me know if you have more questions related to Autism.😊"})

    how_are_you_keywords = ["how are you", "how r u", "how's it going", "how are u"]
    if any(keyword in user_question.lower() for keyword in how_are_you_keywords):
        return jsonify({"reply": "I'm just an Educare bot, but I'm here to help you! 😊 How can I assist you today?"})

    input_embedding = embedding_model.encode(user_question, convert_to_tensor=False)
    scores = torch.tensor([torch.nn.functional.cosine_similarity(
        torch.tensor(input_embedding), torch.tensor(q_emb)).item() for q_emb in question_embeddings])
    best_score = max(scores)
    best_index = scores.index(best_score)

    if best_score < 0.5:
        return jsonify({"reply": "Sorry, I cannot answer that question. You can ask me any question about Autism."})

    return jsonify({"reply": answers[best_index]})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)