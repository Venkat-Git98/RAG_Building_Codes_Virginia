from pinecone import Pinecone

class PineconeManager:
    def __init__(self, api_key, index_name):
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)

    def upsert_vectors(self, vectors_with_ids, batch_size=100):
        vectors_to_upsert = []
        for item in vectors_with_ids:
            vectors_to_upsert.append((
                item['chunk_id'],
                item['embedding'],
                item.get('metadata', {})
            ))
            if len(vectors_to_upsert) >= batch_size:
                self.index.upsert(vectors=vectors_to_upsert)
                vectors_to_upsert = []
        
        if vectors_to_upsert:
            self.index.upsert(vectors=vectors_to_upsert)

    def query_vectors(self, query_vector, top_k=3):
        return self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )