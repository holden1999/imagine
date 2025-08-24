import os
import logging
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv
from config import config
from transformers import BartTokenizer, BartModel
from PIL import Image
import pillow_heif  # Add this import

# Register HEIF opener for PIL
pillow_heif.register_heif_opener()

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageRetriever:
    def __init__(self):
        try:
            db_config = config()
            self.conn = psycopg2.connect(**db_config)
            self.cursor = self.conn.cursor()
            logger.info("PostgreSQL connection initialized")
            # Initialize BART tokenizer and model for feature extraction
            self.bart_tokenizer = BartTokenizer.from_pretrained(
                'facebook/bart-base')
            self.bart_model = BartModel.from_pretrained('facebook/bart-base')
            logger.info(
                "facebook/bart-base tokenizer and model loaded for retrieval embedding search")
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def get_bart_embedding(self, text: str):
        inputs = self.bart_tokenizer(text, return_tensors="pt")
        outputs = self.bart_model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        embedding = last_hidden_states.mean(dim=1).squeeze().tolist()
        return embedding

    def query(self, text_query: str, n_results: int = 5):
        try:
            # Generate embedding for the query using BART
            query_embedding = self.get_bart_embedding(text_query)
            # Vector similarity search on embedding column
            self.cursor.execute(
                """
                SELECT path, metadata->>'caption' AS caption, embedding <-> (%s::vector) AS distance
                FROM media_vectors
                ORDER BY distance ASC
                LIMIT %s
                """,
                (query_embedding, n_results)
            )
            rows = self.cursor.fetchall()
            results = [
                {"path": row[0], "caption": row[1], "distance": row[2]} for row in rows
            ]
            return results
        except Exception as e:
            logger.error(f"Query error: {e}")
            raise

    def display_results(self, results):
        if not results:
            print("No results found.")
            return
        print("\n=== Search Results ===")
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"File: {os.path.basename(result['path'])}")
            print(f"Caption: {result['caption']}")
            print(f"Distance: {result['distance']:.4f}")
            print("-" * 40)
            # Preview image using PIL
            try:
                img = Image.open(result['path'])
                img.show(title=os.path.basename(result['path']))
            except Exception as e:
                print(f"Could not preview image: {e}")


def main():
    retriever = ImageRetriever()
    queries = [
        "yoga",
    ]
    for query in queries:
        print(f"\nSearching for: {query}")
        results = retriever.query(query)
        retriever.display_results(results)


if __name__ == "__main__":
    main()
