import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage ,HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title='Streaming Bot')

st.title("Streaming Bot")

#conversation set-up
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

#get response function
def get_response(query, chat_history):
    template = """You are a help fun assistant, Answer the following question considering the history of the conversation.
    
    Chat_history: {chat_history}
    
    User question: {query}
    """
    prompt = ChatPromptTemplate.from_template(template=template)
    llm = ChatOpenAI()
    
    chain = prompt | llm | StrOutputParser()
    
    #printing a response like streaming
    return chain.stream({
        "chat_history": chat_history,
        "query": query
    })
    
#user input
user_input = st.chat_input("Enter Your Message")
if user_input is not None and user_input!="":
    st.session_state.chat_history.append(HumanMessage(user_input))
    
    with st.chat_message("Human"):
        st.markdown(user_input)
        
    with st.chat_message("AI"):
        #this will return the output same as the chatGPT returning it
        ai_res = st.write_stream(get_response(user_input, st.session_state.chat_history))
        
    st.session_state.chat_history.append(AIMessage(ai_res))