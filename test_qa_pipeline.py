import pytest
import requests
import subprocess
import time
from app.DataLoader import get_document_from_url
from app.answer_evaluating import rerank_by_llm

# to run: pytest test_qa_pipeline.py

# Constants
API_ENDPOINT = "http://localhost:8503/api/v1/question_answering"

@pytest.fixture(scope="module")
def start_server():
    # Start the server
    server_process = subprocess.Popen(["python", "model_server/app.py"])
    
    # Wait for the server to start
    time.sleep(3)
    
    yield
    
    # Stop the server after tests
    server_process.terminate()
    server_process.wait()

def test_qa_pipeline(start_server):
    website_url = "https://pli.princeton.edu/blog/2024/mamba-2-algorithms-and-systems"
    query = "what is means SSD?"
    expected_answer = "structured state space duality"
    # Get context from URL
    context = get_document_from_url(website_url)
    context_content = context[0].page_content

    # Make API call to get response
    response = requests.post(API_ENDPOINT, json={
        "question": query,
        "context": context_content
    })
    
    assert response.status_code == 200, f"API call failed with status code {response.status_code}"
    
    data = response.json()
    responses = data['possible_answers']
    top_response = rerank_by_llm(query, responses)
    print(top_response)
    # Assert that the top response matches the expected answer
    assert top_response.lower() == expected_answer.lower(), f"Expected '{expected_answer}', but got '{top_response}'"