import streamlit as st # type: ignore
from dotenv import load_dotenv # type: ignore
from pypdf import PdfReader # type: ignore
from langchain.text_splitter import CharacterTextSplitter # type: ignore
# from langchain.embeddings import HuggingFaceEmbeddings # type: ignore
# from langchain.vectorstores import Chroma # type: ignore
from langchain_community.embeddings import HuggingFaceEmbeddings # type: ignore
from langchain_community.vectorstores import Chroma # type: ignore

# from langchain.chat_models import ChatOpenAI # type: ignore
# from langchain.memory import ConversationBufferMemory #type: ignore
# from langchain.chains import ConversationlRetrievalChain #type: ignore
import google.generativeai as genai #type: ignore
import os


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

    return vectorstore

# def get_conversation_chain(vectorstore):
#     llm = ChatOpenAI()
#     memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
#     conversation_chain = ConversationlRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory
#     )
#     return conversation_chain

def setup_gemini():
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    return genai.GenerativeModel("gemini-pro")

def ask_gemini(question, vectorstore, model):
    docs = vectorstore.similarity_search(question)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Bạn là chatbot tư vấn tuyển sinh.

    Dựa vào thông tin sau:
    {context}

    Câu hỏi: {question}
    Trả lời:
    """

    response = model.generate_content(prompt)
    return response.text

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with AI", page_icon=":books:")
    st.header("Chat with AI :books:")
    # st.text_input("Ask a question about your documents:")
    # setup model
    model = setup_gemini()

    # input câu hỏi
    user_question = st.text_input("Ask a question about your documents:")

    # hỏi
    if user_question and "vectorstore" in st.session_state:
        answer = ask_gemini(user_question, st.session_state.vectorstore, model)
        st.write(answer)

    with  st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and Click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                # st.write(raw_text)
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)
                # create vector store
                vectorstore = get_vectorstore(text_chunks)
                # st.write(len(text_chunks))
                # st.write("Vector store created")
                # docs = vectorstore.similarity_search("học phí")
                # st.write(docs)

                #create conversation chain
                # 🔥 lưu lại để dùng khi hỏi
                st.session_state.vectorstore = vectorstore

                st.success("Done!")

if __name__ == '__main__':
    main()