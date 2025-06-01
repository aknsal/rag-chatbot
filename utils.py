import re
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that might interfere with processing
    text = re.sub(r'[^\w\s\-.,!?:;"\'()]', ' ', text)
    
    # Remove multiple consecutive punctuation
    text = re.sub(r'[.]{3,}', '...', text)
    
    # Clean up quotes
    text = re.sub(r'["\']{2,}', '"', text)
    
    # Strip and return
    return text.strip()

def chunk_text(text: str, max_chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into chunks with overlap for better context preservation."""
    if not text or len(text) <= max_chunk_size:
        return [text] if text else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_chunk_size
        
        if end < len(text):
            search_start = max(end - 200, start)
            sentence_breaks = []
            
            for pattern in [r'\.\s+', r'\!\s+', r'\?\s+', r'\n\n']:
                for match in re.finditer(pattern, text[search_start:end]):
                    sentence_breaks.append(search_start + match.end())
            
            if sentence_breaks:
                end = max(sentence_breaks)
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = max(end - overlap, start + 1)
        
        if start >= len(text):
            break
    
    return chunks

def extract_tables_from_text(text: str) -> List[str]:
    """Extract table-like structures from text."""
    tables = []
    
    lines = text.split('\n')
    current_table = []
    in_table = False
    
    for line in lines:
        line = line.strip()
        
        separators = ['|', '\t', '  ']
        separator_count = sum(line.count(sep) for sep in separators)
        
        if separator_count >= 2:  # Likely a table row
            current_table.append(line)
            in_table = True
        else:
            if in_table and current_table:
                # End of table
                tables.append('\n'.join(current_table))
                current_table = []
            in_table = False
    
    if current_table:
        tables.append('\n'.join(current_table))
    
    return tables

def format_source_reference(metadata: Dict[str, Any]) -> str:
    """Format source metadata into a readable reference."""
    source = metadata.get('source', 'Unknown')
    title = metadata.get('title', '')
    
    if title and title != source:
        return f"{title} ({source})"
    return source

def validate_openai_key(api_key: str) -> bool:
    """Validate if the OpenAI API key format looks correct."""
    if not api_key:
        return False
    
    return api_key.startswith('sk-') and len(api_key) >= 40

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to be safe for file system."""
    # Remove or replace unsafe characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove excessive underscores
    filename = re.sub(r'_+', '_', filename)
    
    # Strip leading/trailing underscores and spaces
    filename = filename.strip('_ ')
    
    if not filename:
        filename = 'document'
    
    return filename

def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to a maximum length, trying to break at word boundaries."""
    if len(text) <= max_length:
        return text
    
    # Try to break at a word boundary
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # If we can break reasonably close to the limit
        truncated = truncated[:last_space]
    
    return truncated + "..."

def count_tokens_estimate(text: str) -> int:
    """Rough estimate of token count for text (assumes ~4 characters per token)."""
    return len(text) // 4

def is_meaningful_content(text: str, min_length: int = 50) -> bool:
    """Check if text content is meaningful enough to process."""
    if not text or len(text.strip()) < min_length:
        return False
    
    # Check if it's mostly punctuation or special characters
    alphanumeric_chars = sum(c.isalnum() for c in text)
    if alphanumeric_chars < len(text) * 0.3:  # Less than 30% alphanumeric
        return False
    
    return True
