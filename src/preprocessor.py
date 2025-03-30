from bs4 import BeautifulSoup
import re
import logging

logger = logging.getLogger(__name__)

class JobTextPreprocessor:
    def __init__(self, max_length=2000):
        self.max_length = max_length
        
    def clean_html(self, text):
        """Remove HTML tags and clean text"""
        try:
            soup = BeautifulSoup(text, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            return self._normalize_text(text)
        except Exception as e:
            logger.warning(f"Error cleaning HTML: {str(e)}")
            return ""

    def _normalize_text(self, text):
        """Normalize text content"""
        # Remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Truncate and lowercase
        return text[:self.max_length].lower().strip()

    def process_batch(self, df):
        """Process dataframe batch"""
        df['clean_text'] = df['jobDescRaw'].apply(self.clean_html)
        return df