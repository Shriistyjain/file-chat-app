import os
from PyPDF2 import PdfReader
import streamlit as st
from dotenv import load_dotenv
import json
import numpy as np
import faiss

from requests.exceptions import Timeout
from streamlit_extras import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CohereEmbeddings
from langchain.llms import Cohere
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
from langchain.vectorstores import FAISS

# Sidebar contents
load_dotenv()
with st.sidebar:
    st.title('File-Chat-App🫣🌎')
    st.markdown('''
                ## About
                Your Next-Gen Collaborative Platform
                Tired of the hassle of traditional file-sharing methods? Say hello to File-Chat-App, the future of seamless collaboration.
                ''')
    add_vertical_space.add_vertical_space(5)
    st.write('Made with ❤️ by Shristy Jain')

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def main():
    st.header("Chat with PDF")
    
    # Upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    
    if pdf is not None:
        st.write(pdf.name)

        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        
        try:
            # Initialize Cohere API and create embeddings
            cohere_api_key = os.getenv("COHERE_API_KEY")
            if not cohere_api_key:
                st.error("Cohere API key not found. Please set it in your .env file.")
                return
            embeddings_model = CohereEmbeddings(cohere_api_key=cohere_api_key)
            store_name = pdf.name[:-4]
            
            # Try to load the existing vector store from JSON
            if os.path.exists(f"{store_name}.json"):
                with open(f"{store_name}.json", "r") as f:
                    data = json.load(f)
                    vectors = np.array(data["vectors"])
                    dimension = vectors.shape[1]
                    
                    index = faiss.IndexFlatL2(dimension)
                    index.add(vectors)
                    
                    docstore = {i: chunks[i] for i in range(len(chunks))}
                    index_to_docstore_id = {i: i for i in range(len(chunks))}
                    
                    VectorStore = FAISS(index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id, embedding_function=embeddings_model.embed_query)
                    st.write('Embedding loaded from disk')

                    embeddings = vectors

            else:
                # Create a vector store with the text chunks and Cohere embeddings
                embeddings = embeddings_model.embed_documents(chunks)
                embeddings = np.array(embeddings)  # Convert the list to a NumPy array
                dimension = embeddings.shape[1]

                index = faiss.IndexFlatL2(dimension)
                index.add(embeddings)

                
                # Save the vectors to JSON
                with open(f"{store_name}.json", "w") as f:
                    json.dump({"vectors": embeddings.tolist()}, f)
                
                st.write('Vector store created and saved as JSON')
                VectorStore = FAISS(index=index, docstore={i: chunk for i, chunk in enumerate(chunks)}, index_to_docstore_id={i: i for i in range(len(chunks))}, embedding_function=embeddings_model.embed_query)
            
            st.success("PDF successfully processed and embeddings saved!")

            # Query input and similarity search
            query = st.text_input("Ask Questions about PDF")
            if query:
                try:
                    # Embed the query and calculate cosine similarities
                    query_embedding = embeddings_model.embed_query(query)
                    
                    similarities = [cosine_similarity(query_embedding, emb) for emb in np.array(embeddings)]
                    top_indices = np.argsort(similarities)[::-1][:5]  # Get top 5 matches

                    # Get top-matching chunks for QA
                    # matching_chunks = [chunks[idx] for idx in top_indices]  # Define matching_chunks here   
                    matching_chunks = [Document(page_content=chunks[idx]) for idx in top_indices]
                    st.write("Top matching chunks:")
                    for idx in top_indices:
                        st.write(f"Chunk {idx + 1}: {chunks[idx]}")
                        st.write(f"Similarity Score: {similarities[idx]:.2f}")
                        st.write("---")

                    # Use Cohere model for question-answering
                    llm = Cohere(cohere_api_key=cohere_api_key, temperature=0)
                    chain = load_qa_chain(llm=llm, chain_type="stuff")
                    response = chain.run(input_documents=matching_chunks, question=query)
                    st.write(response)

                except Exception as e:
                    st.error(f"Error generating query embedding: {e}")

        except Timeout as e:
            st.error("Request timed out. Please try again.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
