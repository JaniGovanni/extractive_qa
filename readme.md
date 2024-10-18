# Extractive Question Answering System

## Overview

Hello there!
This is my first hosted project on Github. It implements a complete Extractive Question Answering ML pipeline. Although generative QA systems (RAG) are currently receiving the most attention, it is also worth taking a look at extractive QA systems. There have some andvantages over generative QA systems. Also I found the approach quit interesting, because it transforms the task into an classification problem. I implemented the fine-tuning of a DistilBERT model for this task. I mainly oriented me an a course on udemy (Data Science: Transformers for Natural Language Processing) by Lazy Programmer Inc, which I can highly recommend. You can find the code in the training.ipynb notebook. The code might be a little but tricky to understand, but I heavily commented it and its also quit interesting. For building a complete MLOps pipeline, I uploaded the trained model on AWS S3, builded a streamlit interface, builded a flask api for calling the model and Dockerized it all. Further I have deployed it on an AWS EC2 server. When you want to test it, you can contact me, than I will launch the instance. You can also clone the repo and run it locally. It doesnt require much CPU or GPU power. More to this in the following sections.

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
