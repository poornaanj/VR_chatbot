import os
from flask import Flask, render_template, request, jsonify
import subprocess
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from litellm import completion

# Load the embedding model
embedding_model = SentenceTransformer("all-mpnet-base-v2")

# Load the documents and FAISS index
with open('vicRoads_faq.json', 'rt') as f:
    raw_documents = json.load(f)

faiss_index = faiss.read_index("vr_faq_index.idx")

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = os.urandom(16)

@app.route('/')
def index():
    return render_template('index.html')  # Renders the frontend

# Welcome message
@app.route('/welcome', methods=['GET'])
def welcome_message():
    return jsonify({'answer': "Hey, this is VicRoads Chatbot. How can I help you today?"})

# Route to handle chatbot responses
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['question']
    chatbot_response = get_response(user_input)  # Call your chatbot response function
    return jsonify({'answer': chatbot_response})

def get_response(query):
    # Encode the user's query
    query_embedding = embedding_model.encode(query)
    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
    
    # Retrieve relevant documents from FAISS index
    k = 5
    distances, indices = faiss_index.search(query_embedding, k)
    retrieved_documents = [f"{raw_documents[i]['Question']},{raw_documents[i]['Answer']}" for i in indices[0]]
    
    # If no relevant documents are found
    if not retrieved_documents:
        return "Sorry, I couldn't find any relevant information in the documents."

    # Define system and user prompts
 

   
     # Ensure system and user prompts are defined properly
    system_prompt = """You are an assistant specialized in providing accurate information about road licenses in Victoria, Australia. Use the context provided to answer the user's question factually and clearly. 
    Documents: {context}
    """

    user_prompt = """User Question: {question}
    Based on the above documents, please provide a answer, ensuring you include all relevant information. Do not say anything extra.
    If the question has no relevance to the documents, just say 'I don't know.'
    Answer:"""

    # Call the LLM API
    response = completion(
        model="ollama/llama3.2:latest",
        messages=[
            {"content": system_prompt.format(context=retrieved_documents), "role": "system"},
            {"content": user_prompt.format(question=query), "role": "user"}
        ],
        api_base="http://localhost:11434",
        stream=True
    )
    
    response_text = ''  # Initialize an empty string to collect the chunks
    for chunk in response:
        if chunk.choices[0].delta.content:  # Check if there's content in the chunk
            response_text += chunk.choices[0].delta.content  # Append the content to the response_text

    return response_text.strip()

if __name__ == '__main__':
    app.run(debug=True)
