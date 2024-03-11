import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from htmlTemplates import bot_template, css, user_template

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    print("---------------------------------------------------------------------------------------------------------")
    print("Total size: ",len(text)," Bytes")
    print("---------------------------------------------------------------------------------------------------------")
    return text
    
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
    
def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorStore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorStore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userInput(user_input):
    response = st.session_state.conversation({'question': user_input})
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
    st.set_page_config(page_title='Chat with PDFs', page_icon=":books:")
    
    st.write(css, unsafe_allow_html=True)
    
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
        
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("Chat with multiple PDFs")
    user_input = st.text_input('Ask a question about your documents:')
    if user_input:
        handle_userInput(user_input)
    
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            process_pdf_files(pdf_docs=pdf_docs)
            
def process_pdf_files(pdf_docs):
    with st.spinner("Processing"):
        # get pdf text
        raw_text = get_pdf_text(pdf_docs)
        print("from ",pdf_docs," text extracted...............................................................")
        
        # get the text chunks
        text_chunks = get_text_chunks(raw_text)
        # st.write(text_chunks)
        print("chunk created..................................................................................")

        # create vector store
        vectorstore = get_vector_store(text_chunks)
        print("vector store created...........................................................................")

        # create conversation chain
        st.session_state.conversation = get_conversation_chain(vectorstore)
    
    
if __name__ == '__main__':
    main()