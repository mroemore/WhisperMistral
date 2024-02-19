import streamlit as st
import requests as r
import os
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-medium"
client = MistralClient(api_key=api_key)
system_prompt = "You are a helpful AI assistant. You will pay close attention to the level of detail provided in your answer. Your answers should be as short as possible unless the user specifies otherwise. The accuracy of your answers and the quality of your advice is of the utmost importance."

st.title("FuckBot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state['messages'] = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] != "system":
            st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

if len(st.session_state.messages) > 0:
    msgs = []
    for m in st.session_state.messages:
        msgs.append(ChatMessage(role=m['role'], content=m['content']))
    msgs.insert(0, ChatMessage(role='system', content=system_prompt))
    with st.chat_message("assistant"):
        stream = client.chat(
            model=model, 
            messages=msgs
        )
        st.write(stream.choices[0].message.content)
        response = stream.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": response})