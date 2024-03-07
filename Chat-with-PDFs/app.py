import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from htmlTemplates import bot_template, css, user_template

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text
    
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks
    
def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.afrom_documents(text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorStore):
    llm = ChatOpenAI()
    
    template = """You are the bot which helps the user to answer the question in a friendly manner about the PDFs. """
    prompt = PromptTemplate.from_template(template)
    question_generator_chain = LLMChain(llm=llm, prompt=prompt)
    memory = ConversationBufferMemory(memory_key='chat history', return_messages=True)
    retriver = vectorStore.as_retriever()
    
    conversation_chain = ConversationalRetrievalChain.from_llm( question_generator=question_generator_chain, 
                                                               retriever=retriver, 
                                                               memory=memory,
                                                               verbose=True)
    return conversation_chain

async def handle_userInput(user_input):
    response = await st.session_state.conversation({'question': user_input})
    st.session_state.chat_history = response['chat_history']
    
    for i, msg in enumerate(st.sesssion_state.chat_history):
        if i%2==0:
            st.write(user_template.replace("{{MSG}}",msg.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",msg.content), unsafe_allow_html=True)

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
        st.subheader('Your Documents')
        pdf_docs = st.file_uploader("Upload your PDFs and click the 'Process' button",accept_multiple_files=True)
        if st.button('Process'):
            with st.spinner('Processing'):
                
                #get pdf texts
                raw_text = get_pdf_text(pdf_docs)
                
                #get the chunks of the pdfs
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)
                
                #vector store with embeddings
                vectorStore = get_vector_store(text_chunks)
                st.write(vectorStore)
                
                #conversation chain
                st.session_state.conversation = get_conversation_chain(vectorStore)
                
            st.write('Completed')
    
    
if __name__ == '__main__':
    main()