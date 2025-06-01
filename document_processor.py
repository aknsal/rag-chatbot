import os
import re
from typing import List, Dict, Any
import pdfplumber
import pandas as pd
from utils import clean_text, extract_tables_from_text
from icecream import ic

class DocumentProcessor:
    def __init__(self):
        """Initialize the document processor."""
        self.scraped_docs_dir = "scraped_docs"
        os.makedirs(self.scraped_docs_dir, exist_ok=True)
    
    def process_pdf(self, pdf_path: str, filename: str) -> List[Dict[str, Any]]:
        """Process a PDF file and extract text and table content."""
        documents = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                full_text = ""
                tables = []
                
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        full_text += f"\n\nPage {page_num}:\n{page_text}"
                    
                    page_tables = page.extract_tables()
                    for table_num, table in enumerate(page_tables, 1):
                        if table:                            
                            table_text = self._table_to_text(table, page_num, table_num)
                            tables.append(table_text)
                
                if full_text.strip():
                    cleaned_text = clean_text(full_text)
                    documents.append({
                        'content': cleaned_text,
                        'source': filename,
                        'title': f"PDF Document: {filename}",
                        'type': 'pdf_text'
                    })
                
                # Process tables separately
                for table_text in tables:
                    documents.append({
                        'content': table_text,
                        'source': filename,
                        'title': f"Table from {filename}",
                        'type': 'pdf_table'
                    })
        
        except Exception as e:
            raise Exception(f"Error processing PDF {filename}: {str(e)}")
        
        return documents
    
    def _table_to_text(self, table: List[List[str]], page_num: int, table_num: int) -> str:
        """Convert a table to readable text format."""
        if not table:
            return ""
        
        try:
            df = pd.DataFrame(table[1:], columns=table[0])
            df = df.fillna('')
            df = df.astype(str)
            table_text = f"Table {table_num} from Page {page_num}:\n\n"
            headers = " | ".join(df.columns)
            table_text += f"Headers: {headers}\n\n"
            for idx, row in df.iterrows():
                row_text = " | ".join(row.values)
                table_text += f"Row {idx + 1}: {row_text}\n"
            return table_text
        except Exception as e:
            table_text = f"Table {table_num} from Page {page_num}:\n\n"
            for i, row in enumerate(table):
                row_text = " | ".join([str(cell) if cell else "" for cell in row])
                table_text += f"Row {i + 1}: {row_text}\n"
            return table_text
    
    def load_scraped_documents(self) -> List[Dict[str, Any]]:
        """Load documents that were scraped from web pages."""
        documents = []
        
        try:
            for filename in os.listdir(self.scraped_docs_dir):
                ic("Loading file:", filename)
                if filename.endswith('.txt'):
                    file_path = os.path.join(self.scraped_docs_dir, filename)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        if content.strip():
                            title = filename.replace('.txt', '').replace('_', ' ').title()
                            documents.append({
                                'content': clean_text(content),
                                'source': f"Support - {filename}",
                                'title': title,
                                'type': 'web_page'
                            })
                    except Exception as e:
                        print(f"Error loading file {filename}: {str(e)}")
                        continue
        except Exception as e:
            raise Exception(f"Error loading scraped documents: {str(e)}")
        
        return documents
    
    def save_scraped_content(self, content: str, filename: str) -> None:
        """Save scraped content to a file."""
        try:
            file_path = os.path.join(self.scraped_docs_dir, f"{filename}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            raise Exception(f"Error saving scraped content: {str(e)}")
    
    def get_scraped_files_count(self) -> int:
        """Get the number of scraped files."""
        try:
            return len([f for f in os.listdir(self.scraped_docs_dir) if f.endswith('.txt')])
        except:
            return 0
