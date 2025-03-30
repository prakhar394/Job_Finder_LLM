import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class VectorSearchSystem:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.id_map = {}
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
    def create_index(self):
        """Initialize FAISS index"""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        logger.info(f"Created FAISS index with dimension {self.embedding_dim}")

    def generate_embeddings(self, texts, batch_size=256):
        """Generate embeddings in batches"""
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i+batch_size]
            emb = self.model.encode(batch, convert_to_numpy=True)
            embeddings.append(emb)
        return np.vstack(embeddings)

    def add_to_index(self, embeddings, ids):
        """Add embeddings to index with ID mapping"""
        if not self.index:
            self.create_index()
            
        start_idx = self.index.ntotal
        self.index.add(embeddings)
        
        # Update ID mapping
        for i, job_id in enumerate(ids):
            self.id_map[start_idx + i] = job_id
        logger.info(f"Added {len(ids)} embeddings to index")

    def similarity_search(self, query_embedding, k=5, threshold=0.85):
        """Search for similar jobs"""
        distances, indices = self.index.search(query_embedding, k)
        results = []
        for i, (distance, index) in enumerate(zip(distances[0], indices[0])):
            similarity = 1 - distance / 2  # Convert L2 to cosine similarity
            if similarity >= threshold and index in self.id_map:
                results.append({
                    'job_id': self.id_map[index],
                    'similarity': float(similarity)
                })
        return results

    def save_index(self, path):
        """Save FAISS index and mappings"""
        faiss.write_index(self.index, f"{path}/index.faiss")
        np.save(f"{path}/id_map.npy", self.id_map)

    def load_index(self, path):
        """Load FAISS index and mappings"""
        self.index = faiss.read_index(f"{path}/index.faiss")
        self.id_map = np.load(f"{path}/id_map.npy", allow_pickle=True).item()