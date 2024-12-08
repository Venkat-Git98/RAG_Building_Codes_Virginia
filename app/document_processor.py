import fitz
import re
from pathlib import Path
import pymupdf4llm
import pandas as pd
import json
import nltk
from nltk.corpus import stopwords
from generator import generate_embeddings
from pinecone_ops import PineconeManager
from clean import preprocess_md_content
from chunk import (
    process_chunks,
    semantic_refinement,
    sliding_window_chunking,
    extract_metadata_from_filename,
    
)
from config import (
    PROCESSED_CHUNKS_PATH,
    EMBEDDINGS_PATH,
    PINECONE_INDEX_upload_NAME,
    PINECONE_API_KEY
)
import os

class DocumentProcessor:
    def __init__(self, pinecone_index_name=PINECONE_INDEX_upload_NAME):
        self.pinecone_manager = PineconeManager(PINECONE_API_KEY, pinecone_index_name)
        nltk.download('stopwords')
        nltk.download('punkt')
        self.stop_words = set(stopwords.words('english'))

    def _split_pdf_into_sections(self, pdf_path):
        """
    Splits a PDF document into smaller sections based on detected chapters and sections.

    Args:
        pdf_path (str or Path): The path to the PDF file to split.

    Returns:
        Path: The directory containing the split section PDFs.
        """
        output_dir = Path(pdf_path).parent / "split_sections"
        output_dir.mkdir(exist_ok=True)
        
        doc = fitz.open(pdf_path)
        chapter_pattern = re.compile(r'CHAPTER\s+(\d+)', re.IGNORECASE)
        section_pattern = re.compile(r'SECTION\s+(\d+)', re.IGNORECASE)
        
        current_chapter = None
        current_section = None
        current_pages = []
        pdf_count = 0
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            chapter_match = chapter_pattern.search(text)
            section_match = section_pattern.search(text)
            
            if chapter_match or section_match:
                if current_pages:
                    pdf_count += 1
                    self._save_section_to_pdf(doc, current_pages, current_chapter, 
                                           current_section, output_dir, pdf_count)
                    current_pages = []
                
                if chapter_match:
                    current_chapter = chapter_match.group(1)
                if section_match:
                    current_section = section_match.group(1)
                    
            if current_chapter and current_section:
                current_pages.append(page_num)
                
        if current_pages:
            pdf_count += 1
            self._save_section_to_pdf(doc, current_pages, current_chapter, 
                                   current_section, output_dir, pdf_count)
        
        doc.close()
        return output_dir

    def _save_section_to_pdf(self, doc, page_numbers, chapter, section, output_dir, count):
        """
    Saves a list of pages from a PDF document as a new PDF file with metadata in the filename.

    Args:
        doc (fitz.Document): The original PDF document.
        page_numbers (list of int): The page numbers to include in the new PDF.
        chapter (str): The chapter identifier for the section.
        section (str): The section identifier for the section.
        output_dir (Path): The directory to save the new PDF.
        count (int): A unique identifier for the section.
        """
        new_doc = fitz.open()
        for page_num in page_numbers:
            new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
        filename = f"Chapter_{chapter}_Section_{section}_{count}.pdf"
        output_path = output_dir / filename
        new_doc.save(str(output_path))
        new_doc.close()

    def _convert_pdfs_to_markdown(self, pdf_dir):
        """
    Converts PDF files in a directory into Markdown files, including text and table content.

    Args:
        pdf_dir (Path): The directory containing PDF files.

    Returns:
        Path: The directory containing the converted Markdown files.
        """
        markdown_dir = pdf_dir / "markdown"
        markdown_dir.mkdir(exist_ok=True)
        
        for pdf_file in pdf_dir.glob("*.pdf"):
            try:
                result = pymupdf4llm.to_markdown(str(pdf_file), page_chunks=True)
                md_content = ""
                for page in result:
                    md_content += f"{page['text']}\n\n"
                    if 'tables' in page and page['tables']:
                        for table in page['tables']:
                            try:
                                df = pd.DataFrame(table.get('content', []), 
                                                columns=table.get('header', []))
                                md_content += df.to_markdown(index=False) + "\n\n"
                            except Exception as e:
                                print(f"Error processing table: {str(e)}")
                
                output_path = markdown_dir / f"{pdf_file.stem}.md"
                output_path.write_text(md_content, encoding='utf-8')
            except Exception as e:
                print(f"Error processing {pdf_file}: {str(e)}")
                
        return markdown_dir

    def _process_markdown_files(self, markdown_dir):
        """
    Processes Markdown files by cleaning content, extracting metadata, and chunking the text.

    Args:
        markdown_dir (Path): The directory containing Markdown files.

    Returns:
        list of dict: A list of processed chunks with metadata, content, and references.
        """
        initial_chunks = []
        for md_file in markdown_dir.glob("*.md"):
            with open(md_file, 'r', encoding='utf-8') as file:
                content = file.read()
                
            # Clean and preprocess content
            cleaned_content, references = preprocess_md_content(content)
            metadata = extract_metadata_from_filename(md_file.name)
            
            chunk = {
                'content': cleaned_content,
                'references': references,
                'file_name': md_file.name,
                'metadata': metadata
            }
            initial_chunks.append(chunk)
        
        # Process chunks using functions from chunk.py
        processed_chunks = process_chunks(initial_chunks)
        return processed_chunks

    def process_uploaded_document(self, pdf_path):
        """
    Processes an uploaded document by splitting it into sections, converting it to Markdown,
    chunking and refining the text, generating embeddings, and uploading data to Pinecone.

    Args:
        pdf_path (str or Path): The path to the uploaded PDF document.

    Returns:
        tuple: A boolean indicating success or failure and a message describing the result.
        """
        try:
            split_pdfs_dir = self._split_pdf_into_sections(pdf_path)
            print("PDF splitting completed")
            
            markdown_dir = self._convert_pdfs_to_markdown(split_pdfs_dir)
            print("Markdown conversion completed")
            
            chunks = self._process_markdown_files(markdown_dir)
            print("Chunk processing completed")
            
            Path(PROCESSED_CHUNKS_PATH).parent.mkdir(parents=True, exist_ok=True)
            with open(PROCESSED_CHUNKS_PATH, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2)
            print("Chunks saved")
            
            embeddings = generate_embeddings(chunks)
            with open(EMBEDDINGS_PATH, 'w', encoding='utf-8') as f:
                json.dump(embeddings, f, indent=2)
            print("Embeddings saved")
            
            self.pinecone_manager.upsert_vectors(embeddings)
            
            return True, f"Successfully processed {len(chunks)} chunks"
        except Exception as e:
            return False, f"Error processing document: {str(e)}"