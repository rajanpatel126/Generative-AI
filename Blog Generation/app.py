import os
import streamlit as st
from langchain_community.utilities import GoogleSearchAPIWrapper
from dotenv import load_dotenv
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_core.messages import AIMessage ,HumanMessage
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv('HUGGINGFACEHUB_API_TOKEN')
os.environ["GOOGLE_CSE_ID"] = os.getenv('GOOGLE_CSE_ID')
os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')

#google search class fetching data results
class GoogleSearch():
    tool=None
    
    def make_model(self):

        # Tool integration
        def top5_results(query):
            search = GoogleSearchAPIWrapper()
            return search.results(query, 5)

        GoogleSearch.tool = Tool(
            name="Google Search Snippets",
            description="Search Google for top results.",
            func=top5_results,
        )
        
    def ask_query(self,query:str):
        if GoogleSearch.tool is None: 
            self.make_model()
        return GoogleSearch.tool.run(query)

#object of the class
google_obj= GoogleSearch()

#storing the history in the session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

#page confing and title of the page
st.set_page_config(page_title='Blog Generation', page_icon='🤖',)
st.title('Info-GenX Blogs 🤖')

#conversation set-up
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

# Get response from model
def getResponse(query,chat_history):

    # Model calling
    llm = HuggingFaceEndpoint(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",temperature=0.8,max_new_tokens=4096)

    # Prompt Template
    template = """
        You are the writing editor and can write the blog as if you are an experienced writer.

        Write an article on "{query}". Describe the topic as a story with examples that resonate with the reader. Modify the blog as per the user requirements based on the history of chat.
        
        Chat_history = {chat_history}

        Make the blog SEO-friendly.

        If the topic is illegal, harmful, or vulgar, respond with "I cannot assist you with illegal, harmful topics. Seeking information on such topics could indicate harmful intent."
        """

    prompt = ChatPromptTemplate.from_template(template=template)
    chain = prompt | llm | StrOutputParser()
    
    return chain.stream({'chat_history':chat_history , 'query': query})
    
#user input
user_input = st.chat_input("Enter Your Message")
if user_input is not None and user_input!="":
    st.session_state.chat_history.append(HumanMessage(user_input))
    
    with st.chat_message("Human"):
        st.markdown(user_input)
        
    with st.chat_message("AI"):
        for doc in google_obj.ask_query(user_input):
            st.markdown("### "+ doc["title"])
            st.markdown("- "+ doc["link"])
        #this will return the output same as the chatGPT returning it
        ai_res = st.write_stream(getResponse(user_input, st.session_state.chat_history))
        
    st.session_state.chat_history.append(AIMessage(ai_res))



# # Output
# if user_input is not None and user_input!="":
#     with st.spinner("generating..."):
#         # for doc in google_obj.ask_query(input_text):
#         #     st.markdown("### "+ doc["title"])
#         #     st.markdown("- "+ doc["link"])
#         st.markdown(getResponse(user_input, ))
        