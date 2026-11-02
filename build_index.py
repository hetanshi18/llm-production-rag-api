"""Run this LOCALLY (not on Render) to build the search index.

    pip install -r requirements-build.txt
    python build_index.py

It reads the files in data/, splits them into chunks, embeds them and saves
index_embeddings.npy + index_chunks.json. Commit those two files to git.
"""
import os
import json

import numpy as np
from fastembed import TextEmbedding
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = "data"
FILES = [
    ("diabetes.pdf", PyPDFLoader),
    ("dengue.docx", Docx2txtLoader),
    ("mental_health.txt", lambda p: TextLoader(p, encoding="utf-8")),
]

docs = []
for name, loader in FILES:
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        print("MISSING:", path)
        continue
    docs.extend(loader(path).load())

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
chunks = splitter.split_documents(docs)
print("chunks:", len(chunks))

model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
emb = np.array(list(model.embed([c.page_content for c in chunks])), dtype=np.float32)
emb /= np.linalg.norm(emb, axis=1, keepdims=True)

np.save("index_embeddings.npy", emb)
with open("index_chunks.json", "w", encoding="utf-8") as f:
    json.dump(
        [
            {
                "text": c.page_content,
                "source": os.path.basename(c.metadata.get("source", "unknown")),
                "page": c.metadata.get("page"),
            }
            for c in chunks
        ],
        f,
    )
print("saved index_embeddings.npy and index_chunks.json")