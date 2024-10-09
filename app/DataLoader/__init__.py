from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_document_from_url(url):
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()
    return document
