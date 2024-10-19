# Extractive Question Answering System

## Overview

Welcome to one of my first GitHub projects! This repository implements a complete Extractive Question Answering (QA) ML pipeline. While generative QA systems (used inRAG) are currently trending, extractive QA systems offer unique advantages and present an interesting approach by transforming the task into a classification problem.

Key features of this project:
- Fine-tuned DistilBERT model for extractive QA
- Complete MLOps pipeline including:
  - Model training (see `training.ipynb`)
  - Storage of the tuned model in AWS S3 storage
  - Streamlit user interface
  - Flask API for model inference
  - Docker containerization
  - Deployment on AWS EC2

To test the live deployment, please contact me to launch the EC2 instance. Alternatively, you can clone the repository and run it locally with minimal CPU/GPU requirements.

## Project Structure
```
.
├── model_server/
│   ├── app.py
│   └── extractive_qa_answering/
├── app/
│   └── DataLoader/
│   └── answer_evaluating/
│   └── llm/
├── Dockerfile.api
├── Dockerfile.streamlit
├── app_website_chat.py
├── some testing stuff
└── docker-compose.yml
```

## Components

1. **model_server**: This build the flask api and also implements some functions, which are nessecary to run the model in inference. It also contains a little helper function, to download the model from S3.
2. **app/DataLoader**: This implements a basic langchain data loader for websites. I didnt put much effort in it, unlike like in my other project advanced_RAG, which you can also check.
3. **app/answer_evaluating**: This implements some methods, to select the right answer from the model output. More to it in the implementation section.
4. **app/llm**: This loads different models, which are used for the answer_evaluating-methods.
5. **app_website_chat.py**: This implements the streamlit interface.

## Docker Configuration

- `Dockerfile.api`: Sets up the environment for the API server.
- `Dockerfile.streamlit`: Sets up the environment for the Streamlit interface.
- `docker-compose.yml`: Orchestrates the API and Streamlit services.

## Setup and Running

1. Ensure Docker and Docker Compose are installed.
2. Run `docker-compose up --build` in the project root.
3. Access the API at `http://localhost:8503` and the Streamlit interface at `http://localhost:8501`.

Or run the start_app.sh script if you have pulled the repo and installed all dependencies.

## Usage

1. Open the Streamlit interface in a web browser.
2. Enter a URL to load a document.
3. Ask questions about the loaded document.
4. View extracted answers from the system.

## Implementation

The code for calling the model during inference is mostly copied from the training.ipynb, so i didnt explain much here, as its well documented in the source code. Currently i implemented a System to ask question about a website. With little effort this can be expanded to support other document types. Additionally i implemented a simple answer evaluation system, which selects the right answer out of all the answers the model returns, by using a lightweigt llm (llama3.2). You can found the implementation in app/answer_evaluating. This was nessecary, because the model often finds the right answer, as possible answer, but assings a higher probability for a wrong answer. With the additional evaluation system it works ok, but there is more room for improvement. It is worth to mention, that i only trained the model for half an hour on a premium GPU in the collab environment on the SQUAD dataset, so there is defenetly more potential. 

## Dependencies

See the requirements.txt files for details.
