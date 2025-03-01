import os
import chromadb
import logging
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class ImageRetriever:
    def __init__(self):
        """Initialize ImageRetriever with ChromaDB and SentenceTransformer"""
        try:
            # Initialize ChromaDB
            self.chroma_client = chromadb.HttpClient()
            logger.info("ChromaDB client initialized")

            # Get collection
            self.collection = self.chroma_client.get_collection(
                'media_vectors')
            logger.info("Retrieved ChromaDB collection")

            # Initialize SentenceTransformer
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("SentenceTransformer loaded successfully")

        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def create_embeddings(self, text: str) -> list:
        """Create embeddings from text using SentenceTransformer"""
        return self.embedder.encode(text).tolist()

    def query(self, text_query: str, n_results: int = 5) -> dict:
        """Query the collection using text"""
        try:
            logger.info(f"Querying with text: {text_query}")
            query_embedding = self.create_embeddings(text_query)

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )

            logger.info(f"Found {len(results['ids'][0])} matches")
            return results

        except Exception as e:
            logger.error(f"Query error: {e}")
            raise

    def display_results(self, results: dict):
        """Display query results in a formatted way"""
        try:
            if not results['ids'][0]:
                logger.warning("No results found")
                return

            print("\n=== Search Results ===")
            for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
                print(f"\nResult {i}:")
                print(f"Description: {doc}")
                print(f"File path: {meta['path']}")
                print(f"Detections: {meta['detections']}")
                print(f"Object count: {meta['object_count']}")
                print("-" * 50)

        except Exception as e:
            logger.error(f"Error displaying results: {e}")
            raise


def main():
    try:
        # Initialize retriever
        retriever = ImageRetriever()

        # Example queries
        queries = [
            "outdoor scene with trees and sky",
        ]

        # Process each query
        for query in queries:
            print(f"\nSearching for: {query}")
            results = retriever.query(query)
            retriever.display_results(results)

    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()
