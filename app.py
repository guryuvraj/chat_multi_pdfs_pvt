import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import faiss

def get_pdf_text(pdf_docs):
    text = "" #To get the raw text of the pdfs
    for pdf in pdf_docs:

        #PdfReader object for each pdf
        pdf_reader = PdfReader(pdf)

        #To loop through all the pages in each of the pdfs
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text


def get_vectorstore(text_chunks):
    """
    Converts a list of text chunks into a vector store using embeddings.

    This function takes a list of text chunks, converts each chunk into 
    a numerical representation using embeddings, and then stores these 
    embeddings in a vector store for efficient similarity search or 
    other operations.

    """

    # Create an instance of OpenAIEmbeddings.
    # This is likely a class that interfaces with OpenAI's language models
    # to convert text into vector embeddings.
    embeddings = OpenAIEmbeddings()

    # Use the faiss.from_text method to create a vector store.
    # This method takes each text chunk, converts it into an embedding using 
    # the specified embeddings model, and then stores these embeddings in a 
    # vector store. A vector store is a specialized data structure optimized 
    # for fast similarity search and retrieval in large datasets.

    # Parameters for the method are:
    # texts: The list of text chunks to be embedded.
    # embedding: The embedding model to be used for converting text into vectors.
    vectorstore = faiss.from_text(
        texts=text_chunks,
        embedding=embeddings
    )

    return vectorstore

def get_text_chunks(raw_text):
    """
    Splits a given raw text into manageable chunks.

    The function creates an instance of CharacterTextSplitter, 
    a custom class designed to split text based on character count. 
    It then uses this instance to split the provided raw text into smaller chunks.
    """

    # Create a new instance of CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n", #Defines the character used to separate chunks. Here, it's a newline character.
        chunk_size=1000, #The maximum number of characters in each chunk (1000 characters).
        chunk_overlap=200, #The number of characters to overlap between chunks (200 characters).
        length_function=len #The function used to calculate the length of text. Here, it's the built-in len() function.
    )

    # Use the text_splitter to split the provided raw_text into chunks.
    # This will result in a list of text strings, each approximately 1000 characters long,
    # with an overlap of 200 characters between consecutive chunks.
    chunks = text_splitter.split_text(raw_text)

    return chunks




def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:")
    st.header("Chat with multiple PDFs :books:")
    st.text_input("Ask a question about your documents:")

    with st.sidebar:
        st.subheader("Ypur Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process;", 
            accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing"):

                # get the pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)

                # Create vector store
                vecotstore = get_vectorstore(text_chunks)


if __name__ == '__main__':
    main()