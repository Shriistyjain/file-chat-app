# file-chat-app
This README includes a web overview, setup instructions, features, and usage.
File-Chat-App 🫣🌎
File-Chat-App is a next-generation, collaborative platform designed for seamless document interaction. This app allows users to upload PDF files, process them using Cohere embeddings, and ask questions directly about the document content, making document analysis and understanding more interactive.
1.Table of Contents
About the Project
Features
Tech Stack
Getting Started
Prerequisites
Installation
Usage
Configuration
License
Acknowledgments
About the Project
File-Chat-App offers a unique solution to traditional file-sharing and reading methods. By leveraging Cohere's embeddings, FAISS for similarity searches, and an interactive question-answering chain, this app enables users to gain insights from PDF documents more intuitively.

Built with ❤️ by Shristy Jain

2.Features
PDF File Upload: Easily upload and manage PDF documents within the app.
Document Embedding: Efficiently processes documents to generate embeddings using Cohere's API.
Interactive Q&A: Ask questions about the document content, with responses generated based on relevant document sections.
Similarity Search: Retrieve the top relevant chunks of text from the document for precise answers.
Persistent Storage: Saves vectorized data locally, enabling fast access for previously uploaded files.
Tech Stack
Python: Core programming language.
Streamlit: For creating a user-friendly web interface.
Cohere API: Embeddings and language processing.
FAISS: Efficient similarity search and vector storage.
PyPDF2: PDF text extraction.
dotenv: Securely manages environment variables.


3.Getting Started
Prerequisites
Python 3.7+: Ensure Python is installed. You can download it from python.org.
Cohere API Key: Sign up at Cohere to get your API key.
Install dependencies: pip install -r requirements.txt
Installation
Clone the repository:
                git clone https://github.com/yourusername/File-Chat-App.git
                cd File-Chat-App

Install the dependencies:
              pip install -r requirements.txt
Add your Cohere API Key to a .env file:
              COHERE_API_KEY=your_cohere_api_key_here


4.Usage
Run the app:

bash
    streamlit run app.py
Upload and Interact with PDF:

Upload a PDF file via the web interface.
Enter questions about the PDF content in the provided input field.
View top matching document sections and get relevant answers based on your queries.


5.Configuration
In the .env file, specify your Cohere API key:

        COHERE_API_KEY=your_cohere_api_key_here
Ensure .env is in your .gitignore file to keep it secure.

6.Acknowledgments
Cohere for providing language processing APIs.
Streamlit for creating interactive applications with ease.
FAISS by Facebook AI for efficient similarity searches.


