import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

def get_vectorStore_url(url):
    #get the text from the website
    loader = WebBaseLoader(url)
    documents = loader.load()
    
    #split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(documents)
    
    #create the vector store
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())
    return vector_store

def get_contextual_retrival_chain(vector_store):
    llm = ChatOpenAI()
    
    retriver = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriver_chain = create_history_aware_retriever(llm, retriver, prompt)
    
    return retriver_chain

def get_conversational_rag_chain(retriver_chain):
    
    llm = ChatOpenAI()
    
    prompt = ChatPromptTemplate.from_messages([
        ('system', "Answer the User's question based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name='chat_history'),
        ('user', '{input}')
    ])
    stuff_document_chain = create_stuff_documents_chain(llm, prompt)
    
    return create_retrieval_chain(retriver_chain, stuff_document_chain)

def getResponse(user_query):
    retriver_chain = get_contextual_retrival_chain(st.session_state.vector_store)
    
    conversation_rag_chain = get_conversational_rag_chain(retriver_chain)
    
    response = conversation_rag_chain.invoke({
            'chat_history': st.session_state.chat_history,
            'input' : user_query
    })
    
    return response['answer']

# app config
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites")

# sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")
    
if website_url is None or website_url == "":
    st.info("Please Enter the website URL")
else:
    #session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content='Hello, I am a bot. How can I help you?' )
        ]
    #storing vector_store in the session state, so that every time, user doesn't need to upload the website URL.
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = get_vectorStore_url(website_url)
    
    st.info(f"Chatting with {website_url}")
    user_query = st.chat_input("Type your queries here...")
    if user_query is not None and user_query!="":
        response = getResponse(user_query)
        st.session_state.chat_history.append(HumanMessage(user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
        
    #conversations
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
                