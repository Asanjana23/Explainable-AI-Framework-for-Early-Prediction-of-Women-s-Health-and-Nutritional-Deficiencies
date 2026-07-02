import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class RAGService:
    def __init__(self, data_dir="rag_data", model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.data_dir = data_dir
        self.chunks = []
        self.index = None
        self._initialize_index()

    def _initialize_index(self):
        if not os.path.exists(self.data_dir):
            print(f"Warning: {self.data_dir} does not exist.")
            return

        all_text = ""
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(self.data_dir, filename), "r", encoding="utf-8") as f:
                    all_text += f.read() + "\n\n"

        # Simple chunking by double newlines or sentences
        raw_chunks = [c.strip() for c in all_text.split("\n\n") if c.strip()]
        
        # Further split large chunks if needed (very basic)
        self.chunks = []
        for chunk in raw_chunks:
            if len(chunk) > 500:
                # Split by period if too long
                sub_chunks = [s.strip() + "." for s in chunk.split(".") if s.strip()]
                self.chunks.extend(sub_chunks)
            else:
                self.chunks.append(chunk)

        if not self.chunks:
            print("Warning: No text chunks found in rag_data.")
            return

        embeddings = self.model.encode(self.chunks)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        print(f"Indexed {len(self.chunks)} chunks from {self.data_dir}")

    def query(self, text, k=3):
        if self.index is None or not self.chunks:
            return []
        
        query_embedding = self.model.encode([text])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k)
        
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.chunks):
                results.append(self.chunks[idx])
        return results

# Singleton instance
rag_service = RAGService()
