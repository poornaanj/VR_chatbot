import os
from flask import Flask, render_template, request, jsonify
import subprocess
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from litellm import completion

embedding_model = SentenceTransformer("all-mpnet-base-v2")

with open('vicRoads_faq.json', 'rt') as f:
    raw_documents = json.load(f)

# Rename the FAISS index to avoid conflict with any other `index` function
faiss_index = faiss.read_index("vr_faq_index.idx")

app = Flask(__name__)
app.secret_key = os.urandom(16) 


@app.route('/')
def index():
    return render_template('index.html')  # This will render the 'index.html' file in the 'templates' folder

# Route to handle chatbot responses
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['question']
    chatbot_response = get_response(user_input)  # Call your chatbot response function
    return jsonify({'answer': chatbot_response})

def get_response(query):
    query_embedding = embedding_model.encode(query)
    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
    k = 5
    # Now use faiss_index to search
    distances, indices = faiss_index.search(query_embedding, k)
    retrieved_documents = [f"{raw_documents[i]['Question']},{raw_documents[i]['Answer']}" for i in indices[0]]
    
    # Ensure system and user prompts are defined properly
    system_prompt = """You are an assistant specialized in providing accurate information about road licenses in Victoria, Australia. Use the context provided to answer the user's question factually and clearly. 
    Documents: {context}
    """

    user_prompt = """User Question: {question}
    Based on the above documents, please provide a brief answer, ensuring you include all relevant information.Do not say anything that was not asked.\
    If the question asked has no relevance to the documents just say i don't know.

    Answer:"""

    # Call the completion API
    response = completion(
        model="ollama/llama3.2:latest",
        messages=[
            {"content": system_prompt.format(context=retrieved_documents), "role": "system"},
            {"content": user_prompt.format(question=query), "role": "user"}
        ],
        api_base="http://localhost:11434",
        stream=True
    )
    
    response_text = ''
    for chunk in response:
        if chunk.choices[0].delta.content:
            response_text += chunk.choices[0].delta.content+" "

    # Return the collected response text
    return response_text

if __name__ == '__main__':
    app.run(debug=True)
