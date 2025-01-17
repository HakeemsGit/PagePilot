import logging
import asyncio
import aiohttp
import multiprocessing as mp
from urllib.parse import urljoin, urlparse
import re
import unicodedata
import string
from typing import Set, List, Dict, Optional
from bs4 import BeautifulSoup
from concurrent.futures import ProcessPoolExecutor
from functools import partial

class DocumentationScraper:
    def __init__(self, max_concurrent_requests: int = 10):
        """
        Initialize the DocumentationScraper.
        
        Args:
            max_concurrent_requests: Maximum number of concurrent HTTP requests
        """
        self.logger = logging.getLogger(__name__)
        self.visited_urls: Set[str] = set()
        self.doc_urls: List[str] = []
        self.base_url = ""
        self.max_concurrent_requests = max_concurrent_requests
        self.irrelevant_patterns = [
            r'\b(?:contact us|subscribe|related articles)\b',
            r'\b(?:privacy policy|terms of service)\b',
            r'\b(?:copyright|all rights reserved)\b',
        ]
        self.non_char_pattern = re.compile(r'\\u[0-9a-fA-F]{4}|\\[xuU][0-9a-fA-F]{1,6}|[^\w\s.,!?-]')

    def _clean_text(self, text: str) -> str:
        """
        Remove non-characters, Unicode escape sequences, and non-printable characters from text.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text string
        """
        text = text.encode('utf-8').decode('unicode_escape')
        text = self.non_char_pattern.sub('', text)
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(char for char in text if char in string.printable)
        return text.strip()

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """
        Extract clean text from BeautifulSoup object.
        
        Args:
            soup: BeautifulSoup object to extract text from
            
        Returns:
            Extracted and cleaned text
        """
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator=' ', strip=True)
        text = self._clean_text(text)
        return ' '.join(text.split())

    def _is_valid_url(self, url: str) -> bool:
        """
        Check if URL is valid and belongs to the same domain.
        
        Args:
            url: URL to validate
            
        Returns:
            Boolean indicating if URL is valid
        """
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

    async def _fetch_url(self, url: str, session: aiohttp.ClientSession) -> Optional[str]:
        """
        Fetch a single URL asynchronously.
        
        Args:
            url: URL to fetch
            session: aiohttp ClientSession to use
            
        Returns:
            HTML content as string or None if fetch failed
        """
        try:
            self.logger.info(f"Fetching URL: {url}")
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.text()
                self.logger.warning(f"HTTP {response.status} for URL: {url}")
                return None
        except Exception as e:
            self.logger.error(f"Error fetching {url}: {str(e)}")
            return None

    async def _process_chunk(self, urls: List[str]) -> Set[str]:
        """
        Process a chunk of URLs asynchronously.
        
        Args:
            urls: List of URLs to process
            
        Returns:
            Set of discovered URLs
        """
        discovered_urls = set()
        async with aiohttp.ClientSession() as session:
            tasks = []
            for url in urls:
                if url not in self.visited_urls:
                    self.visited_urls.add(url)
                    tasks.append(self._fetch_url(url, session))
            
            responses = await asyncio.gather(*tasks)
            
            for url, html_content in zip(urls, responses):
                if html_content:
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Extract metadata if available
                    meta_description = soup.find('meta', attrs={'name': 'description'})
                    if meta_description:
                        self.logger.info(f"Found metadata: {meta_description.get('content', '')}")
                    
                    # Collect all links
                    for link in soup.find_all('a'):
                        href = link.get('href')
                        if href:
                            full_url = urljoin(url, href)
                            if self._is_valid_url(full_url):
                                discovered_urls.add(full_url)
        
        return discovered_urls

    def _process_with_multiprocessing(self, url_chunks: List[List[str]]) -> Set[str]:
        """
        Process URL chunks using multiple processes.
        
        Args:
            url_chunks: List of URL chunks to process
            
        Returns:
            Set of all discovered URLs
        """
        with ProcessPoolExecutor() as executor:
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_until_complete(self._process_chunk(chunk))
                for chunk in url_chunks
            ]
            return set().union(*tasks)

    def discover_urls(self, base_url: str, chunk_size: int = 10, use_test_urls: bool = False) -> List[str]:
        """
        Discover all documentation URLs from a base URL.
        
        Args:
            base_url: Starting URL for discovery
            chunk_size: Size of URL chunks for parallel processing
            use_test_urls: If True, return test URLs instead of scraping
            
        Returns:
            List of discovered documentation URLs
        """
        if use_test_urls:
            from tests import TEST_URLS
            return TEST_URLS
            
        self.base_url = base_url
        self.visited_urls.clear()
        self.doc_urls.clear()
        
        # Start with the base URL
        initial_urls = {base_url}
        all_discovered_urls = set()
        
        while initial_urls:
            # Process URLs in chunks
            url_chunks = [list(initial_urls)[i:i + chunk_size] 
                         for i in range(0, len(initial_urls), chunk_size)]
            
            # Process chunks with multiprocessing
            new_urls = self._process_with_multiprocessing(url_chunks)
            
            # Update discovered URLs
            all_discovered_urls.update(initial_urls)
            
            # Filter new URLs for next iteration
            initial_urls = {url for url in new_urls 
                          if url not in all_discovered_urls 
                          and self._is_valid_url(url)}
        
        self.doc_urls = sorted(list(all_discovered_urls))
        return self.doc_urls
