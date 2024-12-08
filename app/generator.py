import openai
from tqdm import tqdm
import time
from config import OPENAI_API_KEY, EMBEDDING_MODEL

def add_custom_prompt(text):
    """
    Adds a custom prompt to the input text to improve embedding context.
    """
    prompt = (
        '''Generate a vector embedding for the following text chunk from a building code document. The embedding should:

    * Accurately represent the meaning and context of the text.
    * Capture the relationships between key technical terms and concepts.
    * Emphasize the importance of technical words specific to building codes (e.g., "fire-resistance," "structural integrity," "load-bearing").
    * Prioritize words that convey legal obligations and permissions (e.g., "shall," "must," "may").
    * Be suitable for use in a retrieval-augmented generation (RAG) system where semantic similarity between chunks is crucial.**Details:**\n'''
    )
    return prompt + text

def generate_embeddings(data):
    """
    Generates embeddings for the provided data using OpenAI's API.
    
    Args:
        data: List of dictionaries containing content to embed
        
    Returns:
        List of dictionaries containing chunk_id, embedding, and metadata
    """
    openai.api_key = OPENAI_API_KEY
    embeddings_with_ids = []
    
    for chunk in tqdm(data, desc="Generating embeddings"):
        if not chunk.get('content'):
            print(f"Warning: Empty content for chunk_id: {chunk.get('chunk_id', 'unknown')}")
            continue
            
        try:
            # Add custom prompt to the text
            text = add_custom_prompt(chunk['content'])
            
            # Generate embedding via OpenAI API
            response = openai.Embedding.create(
                input=text,
                model=EMBEDDING_MODEL
            )
            
            # Extract embedding from response
            embedding = response['data'][0]['embedding']
            
            # Store embeddings with ID and metadata
            embeddings_with_ids.append({
                'chunk_id': chunk['chunk_id'],
                'embedding': embedding,
                'metadata': chunk.get('metadata', {})
            })
            
        except openai.error.InvalidRequestError as e:
            print(f"Invalid request for chunk_id {chunk.get('chunk_id', 'unknown')}: {str(e)}")
        except openai.error.RateLimitError:
            print(f"Rate limit exceeded. Waiting before retrying...")
            time.sleep(60)  # Wait for 60 seconds before retrying
        except Exception as e:
            print(f"Unexpected error processing chunk_id {chunk.get('chunk_id', 'unknown')}: {str(e)}")
            continue
    
    print(f"Successfully generated {len(embeddings_with_ids)} embeddings")
    return embeddings_with_ids

def generate_response(prompt):
    """
    Generates a response using OpenAI's GPT model based on the provided prompt.

    Args:
        prompt (str): The input prompt to send to the GPT model.

    Returns:
        str: The generated response from the GPT model or an error message if the 
             response generation fails.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.7,
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error generating response: {str(e)}"
    
def generate_query_embedding(query_text):
    """
    Generates embedding for a query with custom prompt enhancement.
    """
    openai.api_key = OPENAI_API_KEY
    
    # Add custom prompt to query
    enhanced_query = f'''Generate a vector embedding for the following building code query. The embedding should:

    * Match the semantic meaning of queries about building codes and regulations
    * Emphasize technical terms and their relationships
    * Capture the intent of the query in the context of building codes
    * Align with embeddings from building code document chunks
    * Prioritize matching with relevant sections and requirements
    

    Query: {query_text}'''
    
    try:
        response = openai.Embedding.create(
            input=enhanced_query,
            model=EMBEDDING_MODEL
        )
        return response['data'][0]['embedding']
    except openai.error.RateLimitError:
        print("Rate limit exceeded. Waiting before retrying...")
        time.sleep(60)
        return generate_query_embedding(query_text)
    except Exception as e:
        print(f"Error generating query embedding: {str(e)}")
        raise