import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

PDF_FOLDER_PATH = r"publications"  # Specify the path to your folder with PDFs
FAISS_INDEX_PATH = "faiss_index"

def get_pdf_text_from_folder(folder_path):
    text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            pdf_reader = PdfReader(pdf_path)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)

def get_conversational_chain(template):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    chain = LLMChain(llm=model, prompt=prompt)
    return chain

def process_pdfs():
    if not os.path.exists(FAISS_INDEX_PATH):
        with st.spinner("Processing PDFs..."):
            raw_text = get_pdf_text_from_folder(PDF_FOLDER_PATH)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Processing complete and FAISS index created.")

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Enable dangerous deserialization explicitly
    new_db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    # Extracting the context from the documents
    context = "\n".join([doc.page_content for doc in docs])

    # Handling current context question
    current_question_template = """
    What research has been done on the topic asked based on the context provided\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    current_chain = get_conversational_chain(current_question_template)

    current_response = current_chain(
        {"context": context, "question": user_question},
        return_only_outputs=True
    )

    current_response_text = current_response.get("text", "No output text found")

    st.write("Reply: ", current_response_text)

    # Handling future prospects based on the same context and question
    future_question_template = """
    Based on the provided context and the given question, suggest what can be done in the future regarding the topic. Provide detailed and actionable recommendations.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Future Prospects:
    """
    future_chain = get_conversational_chain(future_question_template)

    future_response = future_chain(
        {"context": context, "question": user_question},
        return_only_outputs=True
    )

    future_response_text = future_response.get("text", "No output text found")

    st.write("Future Prospects: ", future_response_text)

def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with your BM Lab💁")

    process_pdfs()

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
