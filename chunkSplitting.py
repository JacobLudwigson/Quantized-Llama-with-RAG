print("Loading Libraries...")
from langchain_community.document_loaders import PyPDFLoader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import json
from typing import List, Dict

print("Loading Embedding Model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
def loadDocument(args):
    folder = args[0]
    name = args[1]
    loader = PyPDFLoader(os.path.join(folder, name))
    pages = loader.load()    

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False
    )
    return text_splitter.split_documents(pages)

def embedDocument(args):
    chunks = args[0]
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = [model.encode(chunk.page_content) for chunk in chunks]
    return embeddings

def searchDocumentForChunkSimilarity(args):
    bestScore = (-90000.0, 0)
    currScore = 0.0
    queryEmbedding = args[0]
    chunkEmbeddings = args[1]
    index = 0
    for chunk in chunkEmbeddings:
        dot = np.dot(chunk, queryEmbedding)
        lenChunkVector = np.linalg.norm(chunk)
        lenQueryVector = np.linalg.norm(queryEmbedding)
        currScore = dot/(lenChunkVector * lenQueryVector)
        if (currScore > bestScore[0]):
            bestScore = (currScore, index)
        index += 1
    return bestScore

def get_response(
    server_url: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 1024,
    stream: bool = True,
) -> str:
    headers = {"Content-Type": "application/json"}
    data = {
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    # Send POST request to the server
    response = requests.post(
        f"{server_url}/v1/chat/completions",
        headers=headers,
        data=json.dumps(data),
        stream=stream,
    )
    response.raise_for_status()  # Ensure the request was successful
    if stream:
        content = ""

        # print("response LINE: ", response.iter_lines())
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8").lstrip("data: ")
                try:
                    json_line = json.loads(decoded_line)
                    if "choices" in json_line and len(json_line["choices"]) > 0:
                        delta = json_line["choices"][0].get("delta", {})
                        content_piece = delta.get("content", "")
                        if "<" in content_piece:
                        # Stop processing when <|im_end|> is encountered
                            break
                        content += content_piece
                        print(content_piece, end="", flush=True)
                except json.JSONDecodeError:
                    continue
        print()  # Ensure the next prompt starts on a new line
        return content
    else:
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            return ""
# Function to run the chatbot
def chatbot(
    server_url: str,
    system_instructions: str = "You are an AI assitant that will assist in answering user questions. Please remember to ask clarifying questions to the user if something is unclear and always be kind.",
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 1024,
    stream: bool = True,
):
    while True:
        print("----------------------------------------Quantized Llama CLI tool----------------------------------------")
        print("1: Chat with Llama without document augmented responses")
        print("2: Chat with Llama WITH document augmented responses")
        print("3: Modify documents being considered in Llama's augmented responses")
        print("4: Upload a pdf to be considered in document augmented responses")
        print("To exit this program, type 'exit' or 'quit' at anytime")
        print("--------------------------------------------------------------------------------------------------------")
        user_input = input("Selection: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        if (str(user_input) == '1'):
            messages = [{"role": "system", "content": system_instructions}]
            while (True):
                prompt = input("User: ")
                userQueryEmbed = model.encode(prompt)
                if prompt.lower() in ["exit", "quit"]:
                    break
                messages.append({"role": "user", "content": prompt})
                print("Assistant: ", end="")
                response = get_response(
                    server_url, messages, temperature, top_p, max_tokens, stream
                )
                messages.append({"role": "assistant", "content": response})
        elif (str(user_input) == '2'):
            while (True):
                prompt = input("User: ")
                userQueryEmbed = model.encode(prompt)
                bestDocumentNumber = 0
                documentNumber = 0
                bestSimScore = (-90000.0, 0)
                print("Performing cosine similarity search on the document set's text chunk embeddings...")
                for document in documents:
                    currSimScore = searchDocumentForChunkSimilarity([userQueryEmbed, documentChunkEmbeddings[documentNumber]])
                    if (currSimScore[0] > bestSimScore[0]):
                        bestSimScore = currSimScore
                        bestDocumentNumber = documentNumber
                    documentNumber += 1
                if prompt.lower() in ["exit", "quit"]:
                    break
                docs = "[DOCUMENT]: '" + chunks[bestDocumentNumber][bestSimScore[1]].page_content + "' [SOURCE]: " + chunks[bestDocumentNumber][bestSimScore[1]].metadata['source'] + " Page number: " + str(chunks[bestDocumentNumber][bestSimScore[1]].metadata['page'])
                print("====================================Fetched Document chunk====================================")
                print(docs)
                print("==============================================================================================")

                system_instructions += "  Please augment your response with this information: " + docs
                messages = [{"role": "system", "content": system_instructions}]
                messages.append({"role": "user", "content": prompt})

                print("Assistant: ", end="")
                response = get_response(
                    server_url, messages, temperature, top_p, max_tokens, stream
                )
                messages.append({"role": "assistant", "content": response})
        elif (str(user_input) == '3'):
            break
        elif (str(user_input) == '4'):
            break
        else:
            print ("Invalid selection!")


documentFolder = "data"
documents = ["GLSL_ES_Specification_3.00.pdf", "Wittgenstein-On Certainty (1).pdf", "caml2012-2.pdf"]

chunks = []
documentChunkEmbeddings = []
print("Loading Documents...")
for i in range (0,len(documents)):
    chunks.append(loadDocument([documentFolder, documents[i]]))
    documentChunkEmbeddings.append(embedDocument([chunks[i]]))

print("Starting chatbot...")
server_url = "http://127.0.0.1:8080"
chatbot(server_url=server_url)