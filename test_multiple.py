from app.DataLoader import get_document_from_url
from app import extractive_qa_answering
from sentence_transformers import CrossEncoder
import numpy as np
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
import re


from app.DataLoader import get_document_from_url
from app import extractive_qa_answering
from sentence_transformers import CrossEncoder
import numpy as np
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from app.answer_evaluating import rerank_by_llm
import re




questions = [
    "What is the capital of Germany?",
    "How many planets are in our solar system?",
    "What is the largest ocean on Earth?",
    "Who painted the Mona Lisa?",
    "What is the chemical symbol for gold?"
]

question = "What is the capital of germany?"

answers = [
    "Berlin",
    "8",
    "Pacific Ocean",
    "Leonardo da Vinci",
    "Au"
]

# score from berlin should be highes
confidence_scores = rerank_by_llm(question, answers)
print(confidence_scores)
