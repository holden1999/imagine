import json
from typing import List
from PIL import Image
from mlx_vlm.utils import load_config
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm import load, generate
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import logging
import psycopg2
from psycopg2.extras import Json
import os
import requests
import base64
from config import config
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class ImageRetriever:
    def __init__(self):
        """Initialize ImageRetriever with PostgreSQL and SentenceTransformer"""
        try:
            # Initialize PostgreSQL connection using config
            db_config = config()
            self.conn = psycopg2.connect(**db_config)
            self.cursor = self.conn.cursor()

            # Initialize embedding model with CPU fallback
            self.embedder = SentenceTransformer(
                'all-MiniLM-L6-v2', device='cpu')

        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def create_embeddings(self, text: str) -> list:
        """Generate text embeddings using CPU-based model"""
        return self.embedder.encode(text).tolist()

    def query(self, text_query: str, n_results: int = 5) -> dict:
        """Query PostgreSQL with Metal-accelerated embeddings"""
        try:
            query_embedding = self.create_embeddings(text_query)
            self.cursor.execute("""
                WITH results AS (
                    SELECT path, metadata ->> 'detections', embedding <-> (%s::vector) AS distance
                    FROM media_vectors
                    WHERE updated_at <= '2025-04-10 17:22:00'::timestamp
                    ORDER BY distance
                    LIMIT %s
                ),
                stats AS (
                    SELECT MIN(distance) as min_dist, MAX(distance) as max_dist 
                    FROM results
                )
                SELECT r.*, 
                    CASE 
                        WHEN distance <= min_dist + (max_dist - min_dist)/3 THEN 'High Match'
                        WHEN distance <= min_dist + 2*(max_dist - min_dist)/3 THEN 'Medium Match'
                        ELSE 'Low Match'
                    END as match_quality
                FROM results r, stats
                ORDER BY distance
            """, (query_embedding, n_results))
            rows = self.cursor.fetchall()

            results = {
                'ids': [[row[0] for row in rows]],
                'metadatas': [[{
                    'path': row[0],
                    'detections': row[1],
                    'distance': row[2],
                    'match_quality': row[3]
                } for row in rows]],
                'description': [[row[1] for row in rows]]
            }
            return results
        except Exception as e:
            logger.error(f"Query error: {e}")
            raise

    def _process_image(self, img: str, detections: list) -> str:
        """Process image using LM Studio API with detections"""
        try:
            with open(img, "rb") as image_file:
                encoded_string = base64.b64encode(
                    image_file.read()).decode('utf-8')

            json_body = {
                "model": "qwen2.5-vl-7b-instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """
answer this question with a structured JSON response:
1. What is this image? add pose if you see a person.
2. "detections" = list all object you see. use specific terms like colour, texture, material, shape.
3. Rewrite descriptive "metadata" image format as dynamic as field you wish and keep 'path' and update 'detections' fields.

Reformat response as:
{
  "result": [
    {
      "detection": answer number 1,
      "metadata": "{"detections": "object1:score|object2:score", "object_count": X, "path": "", "tags": "", "keywords": ""}"
    }
  ]
}
"""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encoded_string}"
                                }
                            },
                            {
                                "type": "text",
                                "text": f"Detections: {detections} path: {img}"
                            }
                        ]
                    }
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "characters",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "result": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "detection": {"type": "string"},
                                            "metadata": {"type": "string"}
                                        }
                                    }
                                }
                            },
                            "required": ["result"]
                        }
                    }
                },
                "temperature": 0.7,
                "max_tokens": -1,
                "stream": False
            }

            response = requests.post(
                "http://127.0.0.1:1234/v1/chat/completions", json=json_body)
            response.raise_for_status()
            return response.json().get("choices", [{}])[0].get("message", {}).get("content", "Description unavailable")

        except Exception as e:
            logger.error(f"Image processing error: {e}")
            return "Description unavailable"

    def display_results(self, results: dict):
        """Display results with vector visualization"""
        try:
            if not results['ids'][0]:
                logger.warning("No results found")
                return

            embeddings = []
            distances = []
            labels = []

            print("\n=== Search Results ===")
            for i, (desc, meta) in enumerate(zip(results['description'][0], results['metadatas'][0]), 1):
                try:
                    embeddings.append(self.create_embeddings(desc))
                    distances.append(meta['distance'])
                    labels.append(os.path.basename(meta['path']))

                    print(f"\nResult {i}:")
                    print(f"Distance Score: {meta['distance']:.3f}")
                    print(f"Description: {desc}")
                    print(f"File: {os.path.basename(meta['path'])}")

                    # Display image and get AI description
                    with Image.open(meta['path']) as img:
                        img.show()

                    # Process single image
                    ai_description = self._process_image(
                        meta['path'], meta['detections'])

                    unmarshall = json.loads(ai_description)
                    metadata = json.loads(unmarshall['result'][0]['metadata'])

                    # Display clean, formatted output
                    print("\nUpdated Information:")
                    print(f"metadata: {metadata}")

                    vector_str = "[" + ",".join(
                        map(str, self.create_embeddings(unmarshall['result'][0]['detection']))) + "]"

                    # Upsert with AI description as document
                    self.cursor.execute("""
                        UPDATE media_vectors
                        SET metadata = %s, embedding = %s::vector, updated_at = NOW()
                        WHERE path = %s
                    """, (unmarshall['result'][0]['metadata'],
                          vector_str,
                          meta['path']))
                    self.conn.commit()

                    print("-" * 50)

                except Exception as e:
                    logger.error(f"Error processing image {i}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error displaying results: {e}")
            raise


def main():

    retriever = ImageRetriever()

    queries = [
        "trees sky sunny",
    ]

    for query in queries:
        print(f"\nSearching for: {query}")
        results = retriever.query(query)
        retriever.display_results(results)


if __name__ == "__main__":
    main()
