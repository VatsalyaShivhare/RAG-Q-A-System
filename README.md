# RAG-Q-A-System
# AmbedkarGPT  (RAG Q&A System)

This project is a small **Retrieval-Augmented Generation (RAG)** system built as an internship assignment.  
It loads an excerpt of **Dr. B. R. Ambedkar’s speech** from `speech.txt`, indexes it using embeddings and a vector database, and then lets you ask natural-language questions about the text via a **command-line interface**.

All model inference runs **locally** using [Ollama](https://ollama.com/) with the **Mistral 7B** model – no API keys or cloud services required.

---

##  Features

- Load document(s) from a local text file (`speech.txt`)
- Split text into **overlapping chunks** for better context
- Create **semantic embeddings** with  
  `sentence-transformers/all-MiniLM-L6-v2`
- Store & retrieve chunks using **ChromaDB** (local vector database)
- Use **Ollama + Mistral 7B** as the LLM
- Simple **CLI Q&A loop**:
  - You type a question
  - System retrieves relevant chunks from the speech
  - LLM answers *grounded in that text*

---

##  Tech Stack

- **Language:** Python 3.11
- **LLM runtime:** [Ollama](https://ollama.com/)
- **Model:** `mistral` (Mistral 7B)
- **Framework:** [LangChain](https://python.langchain.com/)
- **Vector DB:** [ChromaDB](https://www.trychroma.com/)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`

---

## 📂 Project Structure

```text
AmbedkarGPT-Intern-Task/
├─ main.py            # Main RAG pipeline (CLI app)
├─ speech.txt         # Source text (Ambedkar excerpt)
├─ requirements.txt   # Python dependencies
├─ README.md          # Project documentation
└─ chroma_db/         # (Created at runtime) local vector store
