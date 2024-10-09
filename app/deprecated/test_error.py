import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from transformers import pipeline
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
def get_vectorstore_from_url(url):
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()

    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n"],
                                                   chunk_size=200,
                                                   chunk_overlap=50)
    document_chunks = text_splitter.split_documents(document)

    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base",
                                               model_kwargs={"device": "cpu"}
                                               )
    # create a vectorstore from the chunks
    vector_store = Chroma.from_documents(document_chunks, embeddings)
    return vector_store

def get_best_answer(preds,
                    model_inputs,
                    context):

    best_score = float('-inf')
    best_answers = []
    # loop over the splitted sample
    for i, offsets in enumerate(model_inputs['offset_mapping']):
        n_largest = 20
        max_answer_length = 30

        start_logit = preds.start_logits[i]  # (seq_len,) vector
        end_logit = preds.end_logits[i]  # (seq_len,) vector

        # do NOT do the reverse indexing: ['offset_mapping'][idx], much slower

        # get the highest probability tokens
        start_indices = (-start_logit).argsort()[:n_largest]
        end_indices = (-end_logit).argsort()[:n_largest]
        for start_idx in start_indices:
            for end_idx in end_indices:

                # skip answers not contained in context window
                if offsets[start_idx] is None or offsets[end_idx] is None:
                    continue

                # skip answers where end < start
                if end_idx < start_idx:
                    continue

                # skip answers that are too long
                if end_idx - start_idx + 1 > max_answer_length:
                    continue

                # see cell down under for score calculation
                score = start_logit[start_idx] + end_logit[end_idx]
                if score > best_score:
                    best_score = score
                    # find positions of start and end characters
                    first_ch = offsets[start_idx][0]
                    last_ch = offsets[end_idx][1]

                    best_answers.append(context[first_ch:last_ch])
    return best_answers


def get_document_from_url(url):
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()
    document_text = document[0].page_content
    return document_text


def get_question_summary_chain():
    llm = Ollama(model='phi3.5')
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user",
         """Please rephrase the last question from the previous conversation so
          that it incorporates the entire context of our discussion. Return only 1 
          version of the rephrased question and nothing else.""")
    ])

    return LLMChain(llm=llm, prompt=prompt)


def get_summary_question(user_input):
    summary_chain = get_question_summary_chain()

    response = summary_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })

    return response['text']


def get_question_answer(question:str, context:str):
    max_length = 384
    stride = 128
    model_checkpoint = "/Users/jan/Desktop/pdf_extractive_qa/app/models/trained_model_extr_qa_1"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    inputs = tokenizer(
        question,
        context,
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors='pt'
    )
    inputs.pop("overflow_to_sample_mapping")

    # for i in range(len(inputs["input_ids"])):
    #     sequence_ids = inputs.sequence_ids(i)
    #     offset = inputs["offset_mapping"][i]
    #     inputs["offset_mapping"][i] = [
    #         x if sequence_ids[j] == 1 else None for j, x in enumerate(offset)]

    preds = model(inputs['input_ids'], inputs['attention_mask'])
    return get_best_answer(preds, inputs, context)


context = get_document_from_url('https://pli.princeton.edu/blog/2024/mamba-2-algorithms-and-systems')
#vector_store = get_vectorstore_from_url('https://pli.princeton.edu/blog/2024/mamba-2-algorithms-and-systems')
example_question = 'What is much easier to implement than the original selective scan of Mamba?'

#relevant_chunks = vector_store.similarity_search_with_score(query=example_question,
#                                                            k=3)

answers = get_question_answer(question=example_question, context=context)

answers = [answer for answer in answers if answer]
print('I found these possible answers to your question:\n')
for i, answer in enumerate(answers):
    print(f"{i+1}. {answer}\n")


