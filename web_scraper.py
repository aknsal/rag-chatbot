import os
import re
import time
from typing import Tuple, List
import requests
from bs4 import BeautifulSoup
import trafilatura
from urllib.parse import urljoin, urlparse
from document_processor import DocumentProcessor
from readability import Document
from icecream import ic

class AngelOneScraper:
    def __init__(self):
        """Initialize the AngelOne web scraper."""
        self.base_url = "https://www.angelone.in/support"
        self.doc_processor = DocumentProcessor()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.scraped_urls = set()
        
    def scrape_support_pages(self) -> Tuple[bool, str]:
        """Scrape all pages from AngelOne support section."""
        try:
            # Get the main support page
            support_urls = self._discover_support_urls()
            
            if not support_urls:
                return False, "No support URLs found to scrape"
            
            scraped_count = 0
            total_urls = len(support_urls)
            
            for url in support_urls:
                try:
                    success = self._scrape_single_page(url)
                    if success:
                        scraped_count += 1

                    print(f"Scraped {url} ({scraped_count}/{total_urls})")
                    
                    
                except Exception as e:
                    print(f"Error scraping {url}: {str(e)}")
                    continue
            
            if scraped_count > 0:
                return True, f"Successfully scraped {scraped_count} out of {total_urls} support pages"
            else:
                return False, "Failed to scrape any support pages"
                
        except Exception as e:
            return False, f"Error during web scraping: {str(e)}"
    
    def _discover_support_urls(self) -> List[str]:
        """Discover all support URLs from the main support page."""
        urls = set()
        
        try:
            # Get main support page
            response = self.session.get(self.base_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            level1_urls = set()
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(self.base_url, href)
                if self._is_support_url(full_url):
                    level1_urls.add(full_url)
            # Also add the main support page
            level1_urls.add(self.base_url)
            urls.update(level1_urls)  # <-- Ensure all level 1 URLs are included

            # Now, for each level 1 URL, look one level deeper (level 2)
            for url in list(level1_urls):
                try:
                    resp = self.session.get(url, timeout=10)
                    resp.raise_for_status()
                    sub_soup = BeautifulSoup(resp.content, 'html.parser')
                    for link in sub_soup.find_all('a', href=True):
                        href = link['href']
                        full_url = urljoin(self.base_url, href)
                        if self._is_support_url(full_url):
                            urls.add(full_url)
                except Exception as e:
                    print(f"Error discovering sub-URLs in {url}: {str(e)}")
                    continue

            ic(urls)
            
            # Try to find common support page patterns
            common_paths = [
                '/support/faq',
                '/support/trading',
                '/support/account',
                '/support/charges',
                '/support/technical',
                '/support/mobile-app',
                '/support/investment',
                '/support/margin',
                '/support/commodities'
            ]
            
            for path in common_paths:
                test_url = f"https://www.angelone.in{path}"
                try:
                    test_response = self.session.head(test_url, timeout=5)
                    if test_response.status_code == 200:
                        urls.add(test_url)
                except:
                    continue
            
        except Exception as e:
            print(f"Error discovering URLs: {str(e)}")
            # Fallback to just the main support page
            urls.add(self.base_url)
        
        return list(urls)
    
    def _is_support_url(self, url: str) -> bool:
        """Check if a URL is part of the support section."""
        parsed = urlparse(url)
        
        # Must be from angelone.in domain
        if 'angelone.in' not in parsed.netloc:
            return False
        
        # Must contain 'support' in the path
        if 'support' not in parsed.path:
            return False
        
        # Exclude file downloads and external links
        excluded_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx']
        if any(parsed.path.lower().endswith(ext) for ext in excluded_extensions):
            return False
        
        return True
    
    def _scrape_single_page(self, url: str) -> bool:
        """Scrape content from a single page, using Selenium if dynamic content is detected."""
        try:
            if url in self.scraped_urls:
                return True

            response = requests.get(url)
            response.raise_for_status()  # Ensure we notice bad responses

            # Parse the HTML content
            soup = BeautifulSoup(response.text, "html.parser")

            faq_tabs = soup.find_all('div', class_='tab')

            text_content = ""

            for tab in faq_tabs:
                question_label = tab.find('label', class_='tab-label')
                if question_label:
                    question_span = question_label.find('span')
                question = ""
                answer = ""
                if question_span:
                    question = question_span.get_text(strip=True)

                answer_tab = tab.find('div', class_='tab-content')

                if answer_tab:
                    answer_div = answer_tab.find('div', class_='content')
                    
                    if answer_div:
                        answer_parts = answer_div.find_all(['p', 'ol'])
                        ic(answer_parts)
                        for part in answer_parts:
                            if part.name == 'ol':
                                count = 0
                                for li in part.find_all('li'):
                                    count += 1
                                    answer += f"{count}. {li.get_text(strip=True)}\n"
                                answer += "\n"
                            else:
                                answer += part.get_text(strip=True) + "\n"

                if question and answer:
                    text_content += f"Question: {question}\nAnswer: {answer}" + "\n\n"


            if not text_content or len(text_content.strip()) < 100:
                return False

            # Generate filename from URL
            filename = self._url_to_filename(url)

            self.doc_processor.save_scraped_content(text_content, filename)
            self.scraped_urls.add(url)

            return True

        except Exception as e:
            print(f"Error scraping page {url}: {str(e)}")
            return False
    
    def _url_to_filename(self, url: str) -> str:
        """Convert URL to a safe filename."""
        parsed = urlparse(url)
        
        # Extract meaningful parts of the path
        path_parts = [part for part in parsed.path.split('/') if part and part != 'support']
        
        if path_parts:
            filename = '_'.join(path_parts)
        else:
            filename = 'support_main'
        
        # Clean up the filename
        filename = re.sub(r'[^\w\-_]', '_', filename)
        filename = re.sub(r'_+', '_', filename)
        filename = filename.strip('_')
        
        # Ensure it's not empty
        if not filename:
            filename = 'support_page'
        
        return filename
    
    def get_scraped_count(self) -> int:
        """Get the number of pages that have been scraped."""
        return self.doc_processor.get_scraped_files_count()
