import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify
import os
import time
import socket
import torch
from s3_integration import download_dir

from extractive_qa_answering import get_model_and_tokenizer, get_question_answers

app = Flask(__name__)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

####### Load ML Model ##########

model_name = "trained_model_extr_qa_1"
model_checkpoint_path = os.path.join(os.path.dirname(__file__), model_name)

# Check if the model directory exists locally
if not os.path.exists(model_checkpoint_path):
    download_dir(model_checkpoint_path, model_name)

model, tokenizer = get_model_and_tokenizer(model_checkpoint_path)

######## Model Loading ENDS  #############

@app.route("/")
def read_root():
    return f"Hello World: {socket.gethostname()}"

@app.route("/api/v1/question_answering", methods=['POST'])
def question_answering():
    data = request.json
    start = time.time()
    possible_answers = get_question_answers(data['question'], data['context'], model, tokenizer)
    end = time.time()
    prediction_time = int((end-start)*1000)

    output = {
        "model_name": "trained_model_extr_qa_1",
        "question": data['question'],
        "context": data['context'],
        "possible_answers": possible_answers,
        "prediction_time": prediction_time
    }

    return jsonify(output)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8503, debug=True)