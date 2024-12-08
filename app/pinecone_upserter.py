from pinecone import Pinecone
import json
from generator import generate_embeddings
from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PROCESSED_CHUNKS_PATH,
    EMBEDDINGS_PATH
)

class PineconeUpserter:
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(PINECONE_INDEX_NAME)

    def load_chunks(self):
        """
    Loads processed chunks from the JSON file containing embeddings.

    Returns:
        list of dict: A list of chunk dictionaries with embeddings and metadata.
        """
        with open(EMBEDDINGS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)

    def upsert_chunks(self):
        """
    Loads processed chunks, generates embeddings, and upserts them to the Pinecone index.

    Steps:
        1. Loads processed chunks from the embedding JSON file.
        2. Generates embeddings for the chunks.
        3. Prepares vectors with chunk IDs, embeddings, and metadata for upsertion.
        4. Upserts vectors to the Pinecone index in batches.

    Returns:
        tuple: A boolean indicating success or failure and a message with the result.
        """
        try:
            # Load processed chunks
            chunks = self.load_chunks()
            
            # Generate embeddings
            embeddings_with_ids = generate_embeddings(chunks)
            
            # Prepare vectors for upserting
            vectors_to_upsert = [
                (item['chunk_id'], 
                 item['embedding'], 
                 item.get('metadata', {}))
                for item in embeddings_with_ids
            ]
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            return True, f"Successfully upserted {len(vectors_to_upsert)} vectors to Pinecone"
        
        except Exception as e:
            return False, f"Error upserting to Pinecone: {str(e)}"