import os
import logging
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from openai_helper import get_openai_embeddings, get_openai_llm


def get_text_from_pdfs(pdfs):
    logging.info("Inside get_text_from_pdfs")
    text = ""

    for file in pdfs:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_chunks(text, chunk_size, chunk_overlap):
    print("Inside get_chunks")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print("Getting chunks")
    chunks = text_splitter.split_text(text=text)
    return chunks


def process_pdfs(pdfs):
    # print("Inside process_pdfs")
    text = get_text_from_pdfs(pdfs)
    chunks = get_chunks(text, chunk_size=1000, chunk_overlap=0)
    return chunks


def query_pdfs(apikey, docsearch, question):
    llm = get_openai_llm(apikey)
    if question:
        docs = docsearch.invoke(question)
        read_chain = load_qa_chain(llm=llm)
        answer = read_chain.run(input_documents=docs, question=question)
    return(answer)


def get_searchable_docs(apikey, pdfs, process_files):
    chunks = process_pdfs(pdfs)
    embeddings = get_openai_embeddings(apikey)
    docsearch = Chroma.from_texts(chunks, embedding=embeddings).as_retriever()
    return docsearch
