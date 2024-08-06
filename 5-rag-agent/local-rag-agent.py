#Correct with OPENai

import os
from dotenv import load_dotenv
from datetime import datetime
import streamlit as st
import json
from openai import OpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")

openai_client = OpenAI(api_key=openai_api_key)

# Function to call OpenAI API
def call_openai(messages):
    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return completion.choices[0].message.content

# Function to load documents
def load_documents(directory):
    # Load the PDF or txt documents from the directory
    loader = DirectoryLoader(directory)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    return docs

# Function to create a prompt with document context
def create_prompt(docs, user_prompt):
    retrieved_context = " ".join([doc.page_content for doc in docs])
    messages = [
        {"role": "system", "content": "You are a personal assistant who answers questions based on the context provided if the provided context can answer the question."},
        {"role": "system", "content": f"The current date is: {datetime.now().date()}"},
        {"role": "system", "content": f"Context for answering the question:\n{retrieved_context}"},
        {"role": "user", "content": user_prompt}
    ]
    return messages

def main():
    st.title("SEDIVER ALPHA AI ⚡️")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": f"The current date is: {datetime.now().date()}"}
        ]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What questions do you have?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Retrieve documents and create prompt
        docs = load_documents('meeting_notes')
        messages = create_prompt(docs, prompt)

        # Call OpenAI API
        with st.chat_message("assistant"):
            ai_response = call_openai(messages)
            st.markdown(ai_response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": ai_response})

if __name__ == "__main__":
    main()
