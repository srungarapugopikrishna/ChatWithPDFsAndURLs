from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings


def get_openai_llm(apikey):
    llm = OpenAI(temperature=0.6, openai_api_key=apikey)
    return llm


def get_openai_embeddings(apikey):
    embeddings = OpenAIEmbeddings(openai_api_key=apikey)
    return embeddings
