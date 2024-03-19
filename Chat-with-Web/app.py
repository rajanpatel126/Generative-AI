import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever

load_dotenv()
def getResponse(user_query):
    return 'I do not know'

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

# app config
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content='Hello, I am a bot. How can I help you?' )
    ]

# sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")
    
if website_url is None or website_url == "":
    st.info("Please Enter the website URL")
else:
    
    vector_store = get_vectorStore_url(website_url)
    
    retriver_chain = get_contextual_retrival_chain(vector_store)
    
    st.info(f"Chatting with {website_url}")
    user_query = st.chat_input("Type your queries here...")
    if user_query is not None and user_query!="":
        response = getResponse(user_query)
        st.session_state.chat_history.append(HumanMessage(user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
        
        retrived_documents = retriver_chain.invoke({
            'chat_history' : st.session_state.chat_history,
            'input' : user_query
        })
        
        st.write_stream(retrived_documents)
        
    #conversations
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
                