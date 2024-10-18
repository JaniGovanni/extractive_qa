import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from app.DataLoader import get_document_from_url
from app.answer_evaluating import rerank_by_llm
import requests
import os
from app.llm import get_ollama_llm
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

def requests_retry_session(
    retries=3,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
    session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

API_ENDPOINT = os.getenv("API_ENDPOINT", "http://localhost:8503/api/v1/question_answering")

# app config
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites")
# Debug section: Send request to read_root and display response
# try:
#     debug_response = requests.get('http://api:8503/')
#     st.write("Debug: API Root Response")
#     st.write(debug_response.text)
# except Exception as e:
#     st.write(f"Debug: Error connecting to API root: {str(e)}")


# sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

if website_url is None or website_url == "":
    st.info("Please enter a website URL")
else:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, What do you want to know about this website?"),
        ]
    context = get_document_from_url(website_url)
    context_content = context[0].page_content
    # debugging
    # st.write(API_ENDPOINT)
    user_query = st.chat_input("Type your question here...")

    if user_query is not None and user_query != "":
        try:
            with st.spinner("Thinking..."):
                payload = {
                    "question": user_query,
                    "context": context_content
                }
                
                response = requests_retry_session().post(API_ENDPOINT, json=payload, timeout=300)
                #st.write(f"Debug: API Response Status Code: {response.status_code}")
                if response.status_code == 200:
                    response_data = response.json()
                    #st.write("Debug: API Response Data", response_data)
                    answer = response_data['possible_answers']
                    #st.write("Possible Answers:", answer)
                    answer = rerank_by_llm(user_query, answer)
                    #st.write("Reranked Answer:", answer)
                else:
                    st.error(f"Error: API returned status code {response.status_code}")
                    st.write("Response content:", response.text)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=answer))
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)