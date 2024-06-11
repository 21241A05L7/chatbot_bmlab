def list_author_papers(author_name, docs):
    papers = []
    for doc in docs:
        if author_name.lower() in doc.page_content.lower():
            title = doc.metadata.get("title") if doc.metadata and "title" in doc.metadata else "Untitled"
            papers.append(title)
    return "\n".join(papers)

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    if "contribution" in user_question.lower() or "work by" in user_question.lower():
        author_name = user_question.split("by")[-1].strip()
        titles = list_author_papers(author_name, docs)
        if titles:
            st.write(f"Papers by {author_name}:\n" + titles)
            current_response_text = titles
        else:
            st.write(f"No papers found by {author_name}.")
            current_response_text = f"No papers found by {author_name}."
    else:
        context = "\n".join([doc.page_content for doc in docs])

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

    if 'last_question' not in st.session_state or st.session_state.last_question != user_question:
        st.session_state.history.append({"question": user_question, "reply": current_response_text})
        st.session_state.last_question = user_question

    st.button("Copy Answer", on_click=lambda: st.write("Copy the answer manually due to browser restrictions."))
    st.button("Share Answer", on_click=lambda: st.write("Share the answer manually due to browser restrictions."))

    feedback_col1, feedback_col2 = st.columns([1, 1])
    if feedback_col1.button("👍"):
        st.write("Thanks for your feedback!")
    if feedback_col2.button("👎"):
        st.write("Sorry to hear that. Please provide more feedback.")

def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with your BM Lab💁")

    process_pdfs()

    if 'history' not in st.session_state:
        st.session_state.history = []

    user_question = st.text_input('Ask a Question from the PDF Files\nIn the Format:-Summary on your topic')
    if user_question:
        user_input(user_question)

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
