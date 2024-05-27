from url_qna import (get_opneai_llm_object, get_open_ai_embeddings, load_urls,
                     query_chain, split_text, get_faiss_vector_index_from_local,
                     get_faiss_vector_index, save_faiss_vector_index_to_local, get_qa_chain)

from pdf_qna import query_pdfs


def execute_load_urls(urls):
    return load_urls(urls)


def execute_split_text(data):
    return split_text(data)


def execute_vector_index(docs, apikey):
    embeddings = get_open_ai_embeddings(apikey)
    vector_index = get_faiss_vector_index(docs, embeddings)
    save_faiss_vector_index_to_local(vector_index, "faiss_store")


def execute_query(query, apikey):
    llm = get_opneai_llm_object()
    embeddings = get_open_ai_embeddings(apikey)
    vector_index = get_faiss_vector_index_from_local("faiss_store", embeddings)
    chain = get_qa_chain(llm, vector_index)
    result = query_chain(chain, query)
    return result


def execute_query_on_pdfs(api_key, docsearch, question):
    return query_pdfs(apikey=api_key, docsearch=docsearch, question=question)

