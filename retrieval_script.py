from typing import List
from PIL import Image
from mlx_vlm.utils import load_config
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm import load, generate
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import logging
import chromadb
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class ImageRetriever:
    def __init__(self):
        """Initialize ImageRetriever with ChromaDB and SentenceTransformer optimized for Apple Silicon"""
        model_path = "mlx-community/Qwen2.5-VL-7B-Instruct-8bit"
        try:

            # Initialize ChromaDB client
            self.chroma_client = chromadb.HttpClient()
            self.collection = self.chroma_client.get_collection(
                'media_vectors')

            # Initialize embedding model with CPU fallback
            self.embedder = SentenceTransformer(
                'all-MiniLM-L6-v2', device='cpu')

            # Load MLX-optimized vision model with fast processor
            self.model, self.processor = load(model_path, quantized=True)

            self.config = load_config(model_path)

        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def create_embeddings(self, text: str) -> list:
        """Generate text embeddings using CPU-based model"""
        return self.embedder.encode(text).tolist()

    def query(self, text_query: str, n_results: int = 1) -> dict:
        """Query ChromaDB with Metal-accelerated embeddings"""
        try:
            logger.info(f"Querying with text: {text_query}")
            query_embedding = self.create_embeddings(text_query)

            return self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )

        except Exception as e:
            logger.error(f"Query error: {e}")
            raise

    def _process_image(self, img: str) -> str:
        """Process image using proper preprocessing for Qwen VL model"""
        try:
            prompt = "Describe what you see in this image."
            formatted_prompt = apply_chat_template(
                self.processor,
                self.config,
                prompt,
                num_images=1  # Process one image at a time
            )
            response = generate(
                self.model,
                self.processor,
                formatted_prompt,
                [img],  # Pass as single-item list
                verbose=True
            )
            return response

        except Exception as e:
            logger.error(f"Image processing error: {e}")
            return "Description unavailable"

    def display_results(self, results: dict):
        """Display results using Metal-accelerated image processing"""
        try:
            if not results['ids'][0]:
                logger.warning("No results found")
                return

            print("\n=== Search Results ===")
            for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
                try:
                    print(f"\nResult {i}:")
                    print(f"Description: {doc}")
                    print(f"File path: {meta['path']}")

                    # Display image and get AI description immediately
                    with Image.open(meta['path']) as img:
                        img.show()

                    # Process single image
                    ai_description = self._process_image(
                        meta['path'])  # Pass as single-item list

                    print(f"Detections: {meta['detections']}")
                    print(f"Object count: {meta['object_count']}")
                    print(f"AI Description: {ai_description}")
                    print("-" * 50)

                except Exception as e:
                    logger.error(f"Error processing image {i}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error displaying results: {e}")
            raise


def main():
    try:
        retriever = ImageRetriever()

        queries = [
            "outdoor scene with trees and sky"
        ]

        # ,
        # "indoor environment with modern furniture",
        # "cityscape with tall buildings"

        for query in queries:
            print(f"\nSearching for: {query}")
            results = retriever.query(query)
            retriever.display_results(results)

    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()
