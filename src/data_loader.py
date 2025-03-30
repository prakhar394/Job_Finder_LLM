import pandas as pd
import logging

logger = logging.getLogger(__name__)

class JobDataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        
    def load(self):
        """Load and validate job data"""
        try:
            self.df = pd.read_csv(self.file_path, dtype={'lid': str})
            self._validate_data()
            self._clean_data()
            logger.info(f"Loaded {len(self.df)} job postings")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def _validate_data(self):
        """Check required columns"""
        required_cols = {'lid', 'jobTitle', 'companyName', 'jobDescRaw'}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    def _clean_data(self):
        """Basic data cleaning"""
        self.df = self.df.drop_duplicates(subset=['lid'])
        self.df['jobDescRaw'] = self.df['jobDescRaw'].fillna('')