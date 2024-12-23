print("Loading Libraries...")
from langchain_community.document_loaders import PyPDFLoader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import json
import pandas as pd

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
    embeddings = [model.encode(chunk.page_content) for chunk in chunks]
    return embeddings

documentFolder = "data"
documents = os.listdir(documentFolder)

chunks = []
documentChunkEmbeddings = []
print("Loading Documents...")
for i in range (0,len(documents)):
    chunks.append(loadDocument([documentFolder, documents[i]]))
    documentChunkEmbeddings.append(embedDocument([chunks[i]]))

df = pd.DataFrame(columns = ["Chunks", "Chunk Embeddings"])

df["Chunks"] = pd.Series(chunks)
df["Chunk Embeddings"] = documentChunkEmbeddings

df.to_json('docs.json', orient = 'split', compression = 'infer', index = 'true')
print("Documents Loaded!")
