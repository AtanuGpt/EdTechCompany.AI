import streamlit as st
from langchain_helper import create_vector_db, get_qa_chain

st.title("Ed Tech Knowledge Base ☘️")

btn = st.button("Build Knowledgebase")
if btn:    
    create_vector_db()

question = st.text_input("Type your question")

if question:
    chain = get_qa_chain()
    response = chain.run(question)

    st.header("Answer")
    st.write(response)