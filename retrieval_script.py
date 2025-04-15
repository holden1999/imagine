import json
from typing import List
from PIL import Image
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import logging
import psycopg2
from psycopg2.extras import Json
import os
from config import config
import lmstudio as lms
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

    def query(self, text_query: str, n_results: int = 10) -> dict:
        """Query PostgreSQL with Metal-accelerated embeddings"""
        try:
            query_embedding = self.create_embeddings(text_query)
            self.cursor.execute("""
                WITH results AS (
                    SELECT path, metadata ->> 'detections', embedding <-> (%s::vector) AS distance
                    FROM media_vectors
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

    def _get_response_schema(self):
        """Define the JSON schema for image processing response"""
        return {
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

    def _get_image_prompt(self, detections, image_path):
        """Generate the prompt for image analysis"""
        return ("""
            answer this question with a structured JSON response:
            1. Describe this image for embedding vector and add pose if you see a person.
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
            """ + f"Detections: {detections} path: {image_path}")

    def _process_image(self, img: str, detections: list) -> str:
        """Process image using LM Studio API with detections"""
        try:
            with open(img, "rb") as image_path:
                image_handle = lms.prepare_image(image_path)

            model = lms.llm("qwen2.5-vl-7b-instruct")
            chat = lms.Chat()

            schema = self._get_response_schema()
            prompt = self._get_image_prompt(detections, image_path)
            chat.add_user_message(prompt, images=[image_handle])
            prediction = model.respond_stream(chat, response_format=schema)

            # Optionally stream the response
            for fragment in prediction:
                print(fragment.content, end="", flush=True)
            print()

            result = prediction.result()
            desc = result.parsed
            return json.dumps(desc)
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
        ":",
    ]

    for query in queries:
        print(f"\nSearching for: {query}")
        results = retriever.query(query)
        retriever.display_results(results)


if __name__ == "__main__":
    main()
