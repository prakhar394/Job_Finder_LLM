# Job Posting Duplicate Detection System

A system to identify near-duplicate job postings using semantic text embeddings and vector similarity search. Built for scalability and accuracy.

## Features
- **HTML Cleaning**: Robust extraction of text from HTML job descriptions.
- **Text Normalization**: Remove special characters, truncate text, and lowercase conversion.
- **Semantic Embeddings**: Generate 384-dimensional vectors using `all-MiniLM-L6-v2`.
- **FAISS Indexing**: Efficient similarity search for large datasets.
- **Threshold Analysis**: Visualize similarity scores and optimize detection thresholds.

## Installation

### Prerequisites
- Docker and Docker Compose
- Python 3.9+ (optional for local development)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/prakhar394/Job_Finder_LLM.git
   cd Job_Finder_LLM
   
