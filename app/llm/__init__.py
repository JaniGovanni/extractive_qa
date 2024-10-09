from langchain_ollama import ChatOllama
import os
import dotenv
from langchain_groq import ChatGroq
import app.llm

dotenv.load_dotenv()
def get_ollama_llm():
    return ChatOllama(model="llama3.2",
                      temperature=0,
                      base_url=os.getenv('OLLAMA_URL', 'http://localhost:11434'))

def get_groq_llm():
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        #model="llama-3.2-90b-text-preview",
        temperature=0,
        api_key=os.environ.get('GROQ_API_KEY')
    )
    return llm
