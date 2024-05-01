import time
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.llms import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
import streamlit as st

from dotenv import load_dotenv
load_dotenv() 

embeddings = OpenAIEmbeddings()

llm = OpenAI(temperature=0.7)

main_placeholder = st.empty()

def create_vector_db(): 

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=400,
        chunk_overlap=0
    )

    main_placeholder.text("Data Loading...Started...⚙️⚙️⚙️")
    loader = CSVLoader("codebasics_faqs.csv")
    time.sleep(5)

    main_placeholder.text("Text Splitter...Started...⚙️⚙️⚙️")
    data = loader.load_and_split(
        text_splitter=text_splitter
    )
    time.sleep(5)

    main_placeholder.text("Embedding Vector Started Building...⚙️⚙️⚙️")
    db = Chroma.from_documents(
        data,
        embedding=embeddings,
        persist_directory="ChromaEmb"
    )

def get_qa_chain():

    db = Chroma(
        persist_directory="ChromaEmb",
        embedding_function=embeddings
    )

    retriever = db.as_retriever() 

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="map_rerank" 
    )

    return chain





