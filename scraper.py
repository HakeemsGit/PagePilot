import logging
from urllib.parse import urljoin, urlparse
import re
import unicodedata
import string
import requests
from typing import Set, List, Dict, Optional
from bs4 import BeautifulSoup

class DocumentationScraper:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.visited_urls: Set[str] = set()
        self.doc_urls: List[str] = []
        self.base_url = ""
        self.irrelevant_patterns = [
            r'\b(?:contact us|subscribe|related articles)\b',
            r'\b(?:privacy policy|terms of service)\b',
            r'\b(?:copyright|all rights reserved)\b',
        ]
        self.non_char_pattern = re.compile(r'\\u[0-9a-fA-F]{4}|\\[xuU][0-9a-fA-F]{1,6}|[^\w\s.,!?-]')

    def _clean_text(self, text: str) -> str:
        """Remove non-characters, Unicode escape sequences, and non-printable characters from text."""
        text = text.encode('utf-8').decode('unicode_escape')
        text = self.non_char_pattern.sub('', text)
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(char for char in text if char in string.printable)
        return text.strip()

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract clean text from BeautifulSoup object."""
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator=' ', strip=True)
        text = self._clean_text(text)
        return ' '.join(text.split())

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and belongs to the same domain."""
        if not url:
            return False
        
        # Ignore fragments and query parameters
        if '#' in url:
            url = url.split('#')[0]
        if '?' in url:
            url = url.split('?')[0]
            
        try:
            result = urlparse(url)
            base_parsed = urlparse(self.base_url)
            return (all([result.scheme, result.netloc]) and 
                   result.scheme in ['http', 'https'] and 
                   result.netloc == base_parsed.netloc)
        except ValueError:
            return False

    def discover_urls(self, base_url: str) -> List[str]:
        """Discover all documentation URLs from a base URL"""
        self.base_url = base_url
        self.visited_urls.clear()
        self.doc_urls.clear()
        
        def crawl(url: str):
            if url in self.visited_urls:
                return
            
            self.visited_urls.add(url)
            all_page_urls = set()
            
            try:
                self.logger.info(f"Crawling URL: {url}")
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract metadata if available
                meta_description = soup.find('meta', attrs={'name': 'description'})
                if meta_description:
                    self.logger.info(f"Found metadata: {meta_description.get('content', '')}")
                
                # First collect ALL links on the page
                for link in soup.find_all('a'):
                    href = link.get('href')
                    if not href:
                        continue
                    
                    # Convert relative URLs to absolute
                    full_url = urljoin(url, href)
                    all_page_urls.add(full_url)
                
                # Then filter and process valid ones
                for full_url in all_page_urls:
                    if self._is_valid_url(full_url) and full_url not in self.visited_urls:
                        self.doc_urls.append(full_url)
                        crawl(full_url)
                        
            except Exception as e:
                self.logger.error(f"Error crawling {url}: {str(e)}")
        
        crawl(base_url)
        # Remove duplicates and sort for cleaner output
        self.doc_urls = sorted(list(set(self.doc_urls)))
        return self.doc_urls
