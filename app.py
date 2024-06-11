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

def list_paper_titles(user_question, docs):
    # Extracting paper titles based on the user question
    titles = [doc.metadata.get("title", "Untitled") for doc in docs]
    titles_text = "\n".join(titles)
    return titles_text

def list_author_papers(author_name, docs):
    # Extracting paper titles and summaries based on the author name
    papers = []
    for doc in docs:
        if author_name.lower() in doc.page_content.lower():
            title = doc.metadata.get("title", "Untitled")
            summary_template = """
            Provide a summary of the following text:\n\n
            {text}
            """
            summary_chain = get_conversational_chain(summary_template)
            summary_response = summary_chain({"text": doc.page_content}, return_only_outputs=True)
            summary = summary_response.get("text", "No summary found")
            papers.append(f"Title: {title}\nSummary: {summary}\n")
    return "\n".join(papers)

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Enable dangerous deserialization explicitly
    new_db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    # Check if the user is asking for a list of paper titles
    if "list" in user_question.lower() and "titles" in user_question.lower():
        titles_text = list_paper_titles(user_question, docs)
        st.write("Paper Titles:\n", titles_text)
        current_response_text = titles_text
    # Check if the user is asking about work done by a specific author
    elif "work done by" in user_question.lower():
        author_name = user_question.split("work done by")[-1].strip()
        author_papers_text = list_author_papers(author_name, docs)
        st.write(f"Papers by {author_name}:\n", author_papers_text)
        current_response_text = author_papers_text
    else:
        # Extracting the context from the documents
        context = "\n".join([doc.page_content for doc in docs])

        # Handling current context question
        current_question_template = """
        Based on the provided context, describe the research that has been conducted on the given topic.\n\n
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

        st.write("Reply: \n", current_response_text)

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

        st.write("Future Prospects: \n", future_response_text)

    # Add the current question and response to the chat history if it's a new question
    if 'last_question' not in st.session_state or st.session_state.last_question != user_question:
        st.session_state.history.append({"question": user_question, "reply": current_response_text})
        st.session_state.last_question = user_question

    # Adding copy and share buttons
    st.button("Copy Answer", on_click=lambda: st.write("Copy the answer manually due to browser restrictions."))
    st.button("Share Answer", on_click=lambda: st.write("Share the answer manually due to browser restrictions."))

    # Feedback buttons side by side with immediate feedback
    feedback_col1, feedback_col2 = st.columns([1, 1])
    if feedback_col1.button("👍"):
        st.write("Thanks for your feedback!")
    if feedback_col2.button("👎"):
        st.write("Sorry to hear that. Please provide more feedback.")

def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with your BM Lab💁")

    # Process PDFs only if the index does not exist
    process_pdfs()

    # Initialize the session state for chat history if not already done
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Main input for user question
    user_question = st.text_input('Ask a Question from the PDF Files\nIn the Format:-Summary on your topic')
    if user_question:
        user_input(user_question)

    # Display chat history
    st.sidebar.subheader("Chat History")
    if st.session_state.history:
        for i, chat in enumerate(reversed(st.session_state.history[-10:])):
            if st.sidebar.button(f"Topic {i+1}: {chat['question']}", key=f"history_button_{i}"):
                st.session_state.selected_chat = chat

    if 'selected_chat' in st.session_state:
        st.sidebar.write(f"Q: {st.session_state.selected_chat['question']}")
        st.sidebar.write(f"Reply: {st.session_state.selected_chat['reply']}")

if __name__ == "__main__":
    main()
