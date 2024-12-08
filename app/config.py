# config.py
import os
from pathlib import Path

# Base directory configuration
BASE_DIR = Path(__file__).parent
SRC_DIR = BASE_DIR / "src"
APP_DIR = BASE_DIR / "app"
print(BASE_DIR)
# Add src directory to Python path
import sys
sys.path.append(str(SRC_DIR))

# API Keys
OPENAI_API_KEY = "[Open AI API key]"
PINECONE_API_KEY = "[Pinecone API Key]"
DATA_DIR = BASE_DIR / "data"
UPLOADED_CODES_DIR = DATA_DIR / "uploaded_codes"
EXISTING_CODES_DIR = DATA_DIR / "existing_codes"
PROCESSED_CHUNKS_PATH = UPLOADED_CODES_DIR / "processed_chunks.json"
EXISTING_CHUNKS_PATH = EXISTING_CODES_DIR / "existing_chunks.json"


EMBEDDINGS_DIR = DATA_DIR / "embeddings"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "embeddings.json"

# Pinecone settings
PINECONE_INDEX_NAME = "virginia"
PINECONE_INDEX_upload_NAME = "ragbuildingcodesopenaiupload"
# OpenAI settings
EMBEDDING_MODEL = "text-embedding-3-large"
COMPLETION_MODEL = "gpt-4o"

# Create directories if they don't exist
for directory in [DATA_DIR, UPLOADED_CODES_DIR, EMBEDDINGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Function to get absolute path
def get_abs_path(relative_path):
    return BASE_DIR / relative_path

answer_generation_prompt =f"""You are an expert in United States building codes. Armed with sections from these codes, your task is to provide comprehensive and informative answers to user queries. Ensure your responses are well-supported by citing the specific sources of the information.
* **NOTE: If there is no information in the provided context or sections relevant to the query, do not generate details based on your own knowledge. Just rely on the context given.**
"""
Augmenting_prompt = '''
        You are a building code expert assistant tasked with refining queries related to U.S. building codes.
    Focus on enhancing the query by:
    - Identifying and emphasizing main concepts related to building codes.
    - Using specific building code terminology.
    - Including relevant code references where appropriate.
    - Clarifying legal requirements or standards.
    - Keeping geographic specificity accurate.
    - Specifying whether the query seeks definitions, requirements, or procedures.
    * ** NOTE: Do NOT mention different code document in the querry, this model is about the Building codes, so make sure the augmented querry should not provide information about any other codes.
    * ** NOTE: If the user querry is not related to building codes of united states , then say "Given Query is not related to building codes"
   '''
Embediing_prompt = prompt = (
        '''Generate a vector embedding for the following text chunk from a building code document. The embedding should:

    * Accurately represent the meaning and context of the text.
    * Capture the relationships between key technical terms and concepts.
    * Emphasize the importance of technical words specific to building codes (e.g., "fire-resistance," "structural integrity," "load-bearing").
    * Prioritize words that convey legal obligations and permissions (e.g., "shall," "must," "may").
    * Be suitable for use in a retrieval-augmented generation (RAG) system where semantic similarity between chunks is crucial.**Details:**\n'''
    )