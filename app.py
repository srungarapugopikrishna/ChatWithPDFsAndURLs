import streamlit as st

from executor import *
from pdf_qna import get_searchable_docs


def frontend():
    # Streamlit UI

    st.set_page_config(page_title="Chat with Multiple PDF Files", layout="wide")
    st.title("Chat with Multiple PDF files")
    # question = st.text_input("Ask Question Below: ")

    # with st.sidebar:
    #     # st.image("image.jpeg")
    api_key = st.sidebar.text_input("Enter your OpenAI apikey:", placeholder="Enter OpenAI Key", type="password")
    pdfs = st.sidebar.file_uploader("Upload PDF files here", type="pdf", accept_multiple_files=True)
    # process_files_clicked = st.sidebar.button("Process Files")

    urls = []
    no_of_urls = st.sidebar.text_input("No.Of Url Sources: ")
    no_of_urls = 0 if not no_of_urls else int(no_of_urls)
    for i in range(no_of_urls):
        url = st.sidebar.text_input(f"URL {i + 1}::")
        urls.append(url)
    process_url_clicked = st.sidebar.button("Process URLss")
    main_placeholder = st.empty()

    if api_key is not None and process_url_clicked and no_of_urls > 0:
        # print("process_url_clicked:{}".format(process_url_clicked))
        main_placeholder.text("Data Loading....Started.....")
        data = execute_load_urls(urls)
        main_placeholder.text("Text Splitter....Started.....")
        docs = execute_split_text(data)

        execute_vector_index(docs, api_key)

    query = main_placeholder.text_input("Question:")
    if api_key is not None and query and no_of_urls > 0:
        result = execute_query(query, api_key)
        st.header("Answer from URLS")
        st.write(result["answer"])

        sources = result.get("sources")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)
    if api_key is not None and pdfs and query:
        # docsearch = get_searchable_docs(api_key, pdfs, process_files_clicked)
        st.header("Answer from PDFS")
        docsearch = get_searchable_docs(api_key, pdfs, True)
        if docsearch is not None:
            response_from_pdfs = execute_query_on_pdfs(api_key=api_key, docsearch=docsearch, question=query)
            st.write(response_from_pdfs)


if __name__ == "__main__":
    frontend()
