import os
import logging
# from dotenv import load_dotenv
from src.data_loader import JobDataLoader
from src.preprocessor import JobTextPreprocessor
from src.vector_db import VectorSearchSystem
from src.evaluator import DuplicateEvaluator

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4"  # Match CPU cores

logging.basicConfig(level=logging.INFO)

def main():
    # Load data
    loader = JobDataLoader("data/jobs.csv")
    df = loader.load()
    
    # Preprocess text
    preprocessor = JobTextPreprocessor()
    df = preprocessor.process_batch(df)
    
    # Initialize vector system
    vs = VectorSearchSystem()
    vs.create_index()
    
    # Generate and add embeddings
    embeddings = vs.generate_embeddings(df['clean_text'].tolist())
    vs.add_to_index(embeddings, df['lid'].tolist())
    
    # Find duplicates
    all_embeddings = vs.index.reconstruct_n(0, vs.index.ntotal)
    similarities = []
    duplicates = []
    
    for idx in tqdm(range(len(all_embeddings)), desc="Finding duplicates"):
        D, I = vs.index.search(all_embeddings[idx:idx+1], 5)
        for i, distance in zip(I[0], D[0]):
            if idx != i and distance < 0.2:  # Convert to similarity
                similarity = 1 - distance / 2
                duplicates.append({
                    'job_id_1': vs.id_map[idx],
                    'job_id_2': vs.id_map[i],
                    'similarity': similarity
                })
    
    # Save results
    duplicates_df = pd.DataFrame(duplicates)
    duplicates_df.to_csv("results/duplicates.csv", index=False)
    
    # Evaluate
    evaluator = DuplicateEvaluator(duplicates_df['similarity'])
    evaluator.analyze_distribution()
    evaluator.find_optimal_threshold()

if __name__ == "__main__":
    main()