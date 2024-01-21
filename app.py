import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

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
    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

    # st.write(vectorstore)
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



def get_conversation_chain(vectorstore):
    """
    Creates a conversation chain for handling chat interactions.

    This function sets up a conversational retrieval chain that uses a language model 
    for generating responses and a vector store for retrieving relevant information 
    from past interactions. It's particularly useful in chat applications where context 
    and history are important for generating coherent and contextually relevant responses.

    Parameters:
    vectorstore: A vector store containing pre-processed embeddings of text data.

    Returns:
    ConversationalRetrievalChain: An object representing the conversational chain, 
                                  capable of handling chat interactions using the
                                  language model and memory buffer.
    """

    # Create an instance of ChatOpenAI.
    # This is likely a language model provided by OpenAI (such as GPT), 
    # used for generating text-based responses in a conversation.
    # llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})


    # Create an instance of ConversationBufferMemory.
    # This object is responsible for managing the memory of the conversation,
    # typically storing and retrieving the history of the chat.
    # 'memory_keys' specifies the key under which conversation history is stored,
    # and 'return_messages' indicates that the memory should return messages when queried.
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

    # Create a ConversationalRetrievalChain using the language model and vector store.
    # This chain integrates the language model with a retrieval system.
    # The retrieval system is powered by the vectorstore, which enables efficient
    # searching of relevant past conversation pieces.
    # The conversational chain uses these components to generate context-aware responses
    # in a chat application.
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )

    # Return the conversational chain.
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)




def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)




    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

        


    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        handle_userinput(user_question)



    # st.write(user_template.replace("{{MSG}}", "hello bot"), unsafe_allow_html=True)
    # st.write(bot_template.replace("{{MSG}}", "hello human"), unsafe_allow_html=True)



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
                # st.write(text_chunks)

                # Create vector store
                vectorstore = get_vectorstore(text_chunks)

                # Create Conversation chain
                st.session_state.conversation  = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()