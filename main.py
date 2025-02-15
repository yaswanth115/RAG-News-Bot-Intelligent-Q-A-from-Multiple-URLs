import os
import streamlit as st
import pickle
import time
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


# take environment variables from .env (especially openai api key)

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "vector.pkl"

main_placeholder = st.empty()
llm = OllamaLLM(model="llama3.2:1b")

if process_url_clicked:
    valid_urls = [url for url in urls if url]  # Filter out empty strings

    if not valid_urls:  # Check if any valid URLs were entered
        st.warning("Please enter at least one valid URL.")
        st.stop()  # Stop execution if no valid URLs

    loader = UnstructuredURLLoader(urls=valid_urls)  # Use only valid URLs
    main_placeholder.text("Data Loading...Started...✅✅✅")
    try:
        data = loader.load()
    except Exception as e:
        st.error(f"Error loading URLs: {e}") # More informative error message
        st.stop() # Stop further execution if loading fails

    # ... (rest of your code for splitting, embedding, and saving remains the same)

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    # create embeddings and save it to FAISS index
    embeddings = OllamaEmbeddings(model="llama3.2:1b")
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)