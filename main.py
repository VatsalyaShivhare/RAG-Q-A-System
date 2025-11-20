# main.py

import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA


def load_documents(file_path: str):
    """
    Load the speech.txt file into LangChain Document objects.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Make sure it is in the project folder.")
    
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    print(f"[INFO] Loaded {len(documents)} document(s) from {file_path}")
    return documents


def split_documents(documents):
    """
    Split the documents into smaller, overlapping chunks so that
    they fit nicely into the context window and are semantically meaningful.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",    # split on new lines
        chunk_size=300,    # max characters per chunk
        chunk_overlap=50,  # overlap so we don't lose context
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"[INFO] Split into {len(chunks)} chunks")
    return chunks


def create_vectorstore(chunks, persist_directory: str = "chroma_db"):
    """
    Create (or load) a Chroma vector store from the chunks using
    HuggingFace sentence-transformers embeddings.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists(persist_directory):
        print(f"[INFO] Loading existing Chroma DB from {persist_directory}")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
    else:
        print(f"[INFO] Creating new Chroma DB in {persist_directory}")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        vectorstore.persist()
        print("[INFO] Vector store created and persisted to disk.")
    
    return vectorstore


def create_qa_chain(vectorstore):
    """
    Create a RetrievalQA chain that uses:
      - Ollama (Mistral) as the LLM
      - Chroma as the retriever
    """
    llm = Ollama(model="mistral")

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa


def interactive_qa(qa_chain):
    """
    Simple command-line interface to ask questions about the speech.
    """
    print("\n=== AmbedkarGPT Q&A ===")
    print("Ask a question about the speech.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_question = input("You: ")
        if user_question.strip().lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        if not user_question.strip():
            continue
        
        try:
            result = qa_chain({"query": user_question})
            answer = result["result"]
            print(f"\nAssistant: {answer}\n")
        except Exception as e:
            print(f"[ERROR] Something went wrong: {e}")


def main():
    file_path = "speech.txt"
    documents = load_documents(file_path)
    chunks = split_documents(documents)
    vectorstore = create_vectorstore(chunks, persist_directory="chroma_db")
    qa_chain = create_qa_chain(vectorstore)
    interactive_qa(qa_chain)


if __name__ == "__main__":
    main()

