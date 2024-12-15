import requests
import json
from typing import List, Dict
# from sentence_transformers import SentenceTransformer

# # Load a pre-trained embeddings model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Generate an embedding for a sentence
# sentence = "What is the capital of France?"
# embedding = model.encode(sentence)

# print(embedding)  # A dense vector representation

# Function to send a request to the server and get a response
def get_response(
    server_url: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 2048,
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
                            # content += content_piece.split("<|im_end|>")[0]
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
    system_instructions: str = "You are an AI assitant that will assist in document augmented response. Please remember to ask clarifying questions to the user if something is unclear and always be kind.",
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 2048,
    stream: bool = True,
):
    messages = [{"role": "system", "content": system_instructions}]
    while True:
        prompt = input("User: ")
        if prompt.lower() in ["exit", "quit"]:
            break
        messages.append({"role": "user", "content": prompt})
        print("Assistant: ", end="")
        response = get_response(
            server_url, messages, temperature, top_p, max_tokens, stream
        )
        messages.append({"role": "assistant", "content": response})


    

server_url = "http://127.0.0.1:8080"
chatbot(server_url=server_url)