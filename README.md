
# Virginia Building Code Assistant

A Retrieval-Augmented Generation (RAG) system for efficient querying and understanding of Virginia building codes using natural language processing.

---

## Features
- **Natural Language Query Processing**: Understand and search building codes with plain English queries.
- **Dual Search Functionality**: Query Virginia Building Codes or custom document uploads.
- **Semantic Search**: Leverage vector databases for precise and context-aware results.
- **Query Enhancement**: Enhanced results with GPT-4 integration.
- **Interactive Interface**: User-friendly Streamlit interface for seamless interaction.
- **PDF Processing**: Chunking and processing of custom documents for better searchability.

---

## Prerequisites
- Python 3.8+
- Git
- OpenAI API key
- Pinecone API key

---

## Installation

### 1. Clone the Repository
```bash
git clone [your-repository-url]
cd virginia-building-code-assistant
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Additional NLP Requirements
```bash
python -m spacy download en_core_web_sm
python -m nltk.downloader stopwords punkt
```

---

## Configuration

### 1. Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=your_index_name
PINECONE_INDEX_upload_NAME=your_upload_index_name
```

Replace placeholders with your actual API keys and index names.

---

## Project Structure
```
virginia-building-code-assistant/
├── app/
│   ├── main.py              # Streamlit interface
│   ├── document_processor.py # PDF processing pipeline
│   ├── clean.py             # Text cleaning utilities
│   ├── chunk.py             # Chunking logic
│   ├── generator.py         # Embedding generation
│   ├── pinecone_ops.py      # Vector database operations
│   └── config.py            # Configuration settings
├── data/
│   ├── existing_codes/      # Processed existing codes
│   └── embeddings/          # Generated embeddings
├── requirements.txt
└── .env
```

---

## Usage

### 1. Start the Application
Run the Streamlit app:
```bash
streamlit run app/main.py
```

### 2. Using the Interface
- **Select Search Source**: Choose "Virginia Building Codes" or "Upload Documents."
- **For Custom Documents**:
  - Upload a PDF through the sidebar.
  - Click **Process Document**.
  - Wait for processing completion.
- **Enter Query**: Use the text input box to enter a query.
- **View Results**:
  - Enhanced query.
  - Relevant building code sections.
  - AI-generated response.

---

## Error Handling

### Cleanup Function
- Use the **Cleanup** button in the sidebar to:
  - Remove temporary files.
  - Clear vector database.
  - Resolve permission issues.

### Common Issues
- **OpenAI API Errors**: Check your API key and rate limits.
- **Pinecone Errors**: Ensure correct API key and index configuration.

---

## Dependencies
```text
streamlit==1.28.0
openai==1.3.0
pinecone-client==2.2.4
PyMuPDF==1.23.5
pymupdf4llm==0.1.1
pandas==2.1.1
nltk==3.8.1
spacy==3.7.2
python-dotenv==1.0.0
tqdm==4.66.1
```


Copy and save this as `README.md` in your project directory. Replace placeholders like `[your-repository-url]`, `your_openai_key`, and `your_pinecone_key` with actual values. Add your license and contributor information as needed.
