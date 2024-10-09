from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
# import re
from sentence_transformers import CrossEncoder
import numpy as np
from app.llm import get_groq_llm, get_ollama_llm


# this could be adjusted in such a way, that the llm has to choose
# 1 possible answer instead of give scores
def rerank_by_llm(question, answers):
    """
    Asses, if an answer corresponds to a question or not by an llm.
    """
    llm = get_ollama_llm()

    system_template = """
    "You are a helpful assistant that evaluates whether a given answer matches a given question. 
    You assess whether the answer correctly responds to the question. 
    To make this assessment, you return a value between 0 and 1. 
    The closer the value is to 1, the more confident you are that the answer correctly answers the question. 
    If you believe the answer does not match the question, you return a value closer to 0. Only response with a value
    and nothing else in your answer. I'm only interested in the value you assign, so please just provide the score.
    Please only provide a score of 1, if you are absolutely sure, that die answers answers the given question."
    Here are a few examples:

    Question: "Who is the current president of peru?"
    Answer: "Dina Boluarte"
    AIRespond: 0.9

    Question: "What means NSHR?"
    Answer: "National Society for Human Rights"
    AIRespond: 0.8

    Question: "Who was Adolf Hitler?"
    Answer: "A singer"
    AIRespond: 0.15
    """

    input_template = """
    Question:{query}
    Answer: {answer}
    AIRespond:
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system", system_template,
            ),
            ("human", input_template),
        ]
    )
    pairs = [[question, doc] for doc in answers]

    chain = {"query": lambda x: x[0], "answer": lambda x: x[1]} | prompt | llm
    response = chain.batch(pairs, {"max_concurrency": 5})
    # deprecated. Nessecary for phi3.5
    # match = [re.search(r"[-+]?\d*\.\d+|\d+", resp.content).group(0) for resp in response]
    # best_answer_ind = match.index(max(match))
    values = [float(resp.content) for resp in response]
    best_answer_ind = values.index(max(values))
    return answers[best_answer_ind]

def rerank_by_crossencoder(answers, original_query, top_k=2):
    """
    Uses an crossencoder (from sentence-transformers), to determine if the answer
    is connected to the query. After this, the answers, which are
    most connected to the original query gets returned
    :param answers: possible answers to the query
    :param original_query: The original user query
    :param top_k: How many answers should be returned
    :return: top_k to the query most connected answers
    """

    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [[original_query, doc] for doc in answers]
    scores = cross_encoder.predict(pairs)

    top_indices = np.argsort(scores)[::-1][:top_k]
    top_answers = [answers[i] for i in top_indices]
    return top_answers