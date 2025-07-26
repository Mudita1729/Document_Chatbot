# DocuMentor AI: Your Personal Document Q&A Expert

DocuMentor AI is a powerful and intuitive web application that allows you to "chat" with your documents. Upload PDFs, Word documents, or text files, and ask complex questions to receive instant, accurate, and source-cited answers.

## ✨ Key Features

*   **Multi-Format Support:** Ingests and processes `.pdf`, `.docx`, and `.txt` files.
*   **AI-Powered Answers:** Leverages the speed of the Groq Llama 3 model to understand context and generate human-like answers.
*   **Grounded & Cited Responses:** Eliminates hallucination by basing answers *only* on the content of your documents and citing the source file(s).
*   **High-Performance Search:** Uses `FAISS` vector search for lightning-fast retrieval of relevant information, even from large documents.
*   **Intuitive UI:** A clean, professional, and easy-to-use interface built with Streamlit.
*   **Simple & Local:** Runs locally. Your documents are processed on your machine and are not uploaded to any third-party server (only the relevant text snippets are sent to the Groq API for answer generation).

## 🛠️ How It Works (Tech Stack)

This application is built on the **Retrieval-Augmented Generation (RAG)** architecture.

1.  **Document Ingestion & Chunking:** Documents are loaded, and their text is extracted (`PyPDF2`, `python-docx`) and then broken down into smaller, manageable chunks.
2.  **Embedding:** Each text chunk is converted into a numerical vector representation (an "embedding") using a `sentence-transformers` model.
3.  **Indexing & Storage:** The embeddings are stored in a `FAISS` vector index, which is highly optimized for fast similarity searches.
4.  **Retrieval:** When you ask a question, it's also converted into an embedding. FAISS then efficiently finds the most relevant text chunks from your documents based on vector similarity.
5.  **Generation:** The original question and the retrieved text chunks are passed as context to the `Groq` Llama 3 model, which generates a final, coherent answer based *only* on that context.

## 🚀 Setup and Installation

Follow these steps to get the application running on your local machine.

### Prerequisites

*   Python 3.8+
*   An API key from [GroqCloud](https://console.groq.com/keys)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/documentor-ai.git
cd documentor-ai
```

### 2. Create and Activate a Virtual Environment

*   **Windows:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
*   **macOS / Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

### 3. Install Dependencies

All required packages are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 4. Set Up Your API Key

The application needs your Groq API key. You can set it up in one of two ways:

*   **(Recommended) Create a `.env` file:**
    1.  Create a file named `.env` in the root of the project folder.
    2.  Add your API key to it:
        ```
        GROQ_API_KEY="gsk_YourSecretKeyHere"
        ```
    3.  The application will automatically load this key. **Remember to add `.env` to your `.gitignore` file to keep your key secret!**

*   **(Alternative) Use the UI:**
    If no environment variable is found, you can paste your API key directly into the input box in the app's sidebar.

### 5. Run the Application

```bash
streamlit run app.py
```

Your web browser should automatically open with the application running!

## 📖 How to Use

1.  **Launch the app** using the command above.
2.  **Enter your Groq API key** in the sidebar if you haven't set up a `.env` file.
3.  **Upload your documents** using the file uploader, or provide a path to a local directory containing your files.
4.  **Click "Load & Process Documents"**.
5.  Once processing is complete, **ask any question** about your documents in the main input box and get your answer!
