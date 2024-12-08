import os
import re
import json
import pathlib
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
phrases_to_remove = [
    "# # copyright 2024 international code council , inc. , licensors ( rights reserved ) . accessed venkatesh shanmugam",
    "pursuant license agreement icc . reproduction distribution authorized .",
    "unauthorized reproduction distribution violation federal copyright , subject civil -- -- -"
]

# Define stopwords that should be retained for context-specific reasons
retained_stopwords = {
    'shall', 'must', 'may', 'should', 'section', 'chapter', 'article', 'table',
    'inch', 'inches', 'feet', 'foot', 'mm', 'psi', 'mpa'
}

# Get the default set of English stopwords and remove the retained ones
default_stopwords = set(stopwords.words('english'))
custom_stopwords = default_stopwords.difference(retained_stopwords)

# Define the copyright pattern with improved flexibility
COPYRIGHT_PATTERN = re.compile(
    r'##\s+copyright\s+\d{4}\s+international\s+code\s+council\s*,\s*'
    r'inc\.\s*,\s*licensors\s*\([^)]*\)\s*\.\s*accessed\s+.*?\d{1,2}/\d{1,2}/\d{4}\s+'
    r'pursuant\s+to\s+license\s+agreement\s+with\s+icc\s*\.\s*reproduction\s+and\s+'
    r'distribution\s+authorized\s*\.\s*unauthorized\s+reproduction\s+and\s+'
    r'distribution\s+is\s+a\s+violation\s+of\s+federal\s+copyright\s*,\s*subject\s+to\s+'
    r'civil\s+and\s+criminal\s+penalties\s*\.*\s*-+',
    re.DOTALL | re.IGNORECASE
)


# Define patterns for removing hyperlinks
LINK_PATTERNS = [
    r'\[([^\]]+)\]\(http[s]?://[^\)]+\)',  # Convert markdown links to text
    r'\(http[s]?://[^\)]+\)',              # Remove plain URL in parenthesis
    r'http[s]?://\S+',                     # Remove standalone URLs
    r'\[http[s]?://[^\]]+\]',              # Remove markdown URLs without display text
    r'_Accessed by.*?thereunder_',         # Remove access information
    r'_+'                                  # Remove extra underscores
]

# Define a pattern for extracting references based on certain keywords
REFERENCE_PATTERN = r'(see|accordance with|comply with|specified in|determined by|defined in|subject to|listed in)\s+((?:Section|Chapter|Table|Figure)\s+[\d\.]+(?:\s*(?:through|to)\s*[\d\.]+)?|\b[A-Z]+\s+[\d\.]+(?:\s*(?:through|to)\s*[\d\.]+)?)'

# Function to remove special character encodings
def clean_special_characters(text):
    text = text.encode('utf-8').decode('unicode_escape')
    text = re.sub(r'[^a-zA-Z0-9 ,.?!]', '', text)
    text = re.sub(r'[^\x20-\x7E]+', '', text)  # Keeps standard ASCII characters
    return text

# Function to remove copyright and unnecessary content
def remove_copyright_and_links(content):
    content = COPYRIGHT_PATTERN.sub('', content)
    for pattern in LINK_PATTERNS:
        content = re.sub(pattern, '', content)
    return content.strip()

# Function to remove specific phrases from the content
def remove_phrases(content, phrases):
    for phrase in phrases:
        content = content.replace(phrase, '')
    return content

# Function to extract references using a predefined pattern
def extract_references(content):
    return re.findall(REFERENCE_PATTERN, content, re.IGNORECASE)

# Function to preprocess Markdown content and extract references
def preprocess_md_content(content):
    # Extract references first
    references = extract_references(content)
    
    # Convert to lower case
    content = content.lower()
    content = remove_phrases(content, phrases_to_remove)
    # General cleanup
    content = re.sub(r'^(CHAPTER|SECTION)\s+\d+.*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^(Page \d+|Confidential|Do Not Distribute)$', '', content, flags=re.MULTILINE)
    content = re.sub(r'\bPage\s+\d+\b', '', content)
    content = re.sub(r'\s+', ' ', content).strip()
    content = re.sub(r'\*+', ' ', content)
    content = clean_special_characters(content)
    content = remove_copyright_and_links(content)

    # Tokenize and remove custom stopwords
    tokens = word_tokenize(content)
    filtered_tokens = [token for token in tokens if token not in custom_stopwords]

    # Prepare the cleaned content
    cleaned_content = ' '.join(filtered_tokens)

    # Simplify references to a comma-separated list
    references = ', '.join(set(ref[1] for ref in references))

    return cleaned_content, references




