import unittest
from scraper import DocumentationScraper
from test_data import TEST_URLS

class TestDocumentationAssistant(unittest.TestCase):
    def setUp(self):
        self.scraper = DocumentationScraper()
    
    def test_url_discovery(self):
        """Test URL discovery with pre-selected links"""
        for url in TEST_URLS[:3]:  # Test first 3 URLs
            urls = self.scraper.discover_urls(url)
            self.assertTrue(len(urls) > 0)
            self.assertTrue(url in urls)
    
    def test_query_response(self):
        """Test basic query functionality"""
        question = "How do I use unittest in Python?"
        answer, sources = query_docs(question)
        self.assertIsNotNone(answer)
        self.assertIsNotNone(sources)

if __name__ == '__main__':
    unittest.main()
