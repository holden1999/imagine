import os
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import psycopg2
from psycopg2.extras import Json
import logging
from dotenv import load_dotenv
import time
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MediaIndexer:
    def __init__(self, model_path: str):
        """Initialize the indexer with YOLO model, SentenceTransformer and pgvector"""
        try:
            # Initialize YOLO
            self.model = YOLO(model_path)
            logger.info("YOLO model loaded successfully")

            # Initialize SentenceTransformer
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("SentenceTransformer loaded successfully")

            # Initialize PostgreSQL connection
            self.conn = psycopg2.connect(
                host=os.getenv('POSTGRES_HOST'),
                port=os.getenv('POSTGRES_PORT'),
                user=os.getenv('POSTGRES_USER'),
                password=os.getenv('POSTGRES_PASSWORD'),
                dbname=os.getenv('POSTGRES_DB')
            )
            self.cursor = self.conn.cursor()
            logger.info("PostgreSQL connection initialized")
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def create_embeddings(self, text: str) -> list:
        """Create embeddings from text using SentenceTransformer"""
        return self.embedder.encode(text).tolist()

    def _is_valid_image(self, file_path: str) -> bool:
        """Check if the image file is valid and readable"""
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                img.verify()
            return True
        except Exception:
            return False

    def index_media(self, directory: str):
        """Index all media files in the directory"""
        try:
            # Clear terminal
            os.system('clear' if os.name != 'nt' else 'cls')
            logger.info(f"Starting media indexing in directory: {directory}")

            # Collect and validate image paths
            image_paths = []
            skipped_files = []
            print("\nValidating images...")
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                        file_path = os.path.join(root, file)
                        if self._is_valid_image(file_path):
                            image_paths.append(file_path)
                        else:
                            skipped_files.append(file_path)
                            logger.warning(
                                f"Skipping invalid image: {file_path}")

            if skipped_files:
                print(f"\nSkipped {len(skipped_files)} invalid images")

            if not image_paths:
                logger.warning(
                    f"No valid images found in directory: {directory}")
                return

            total_files = len(image_paths)
            total_batches = (total_files + 31) // 32
            processed_files = 0

            print(
                f"\nFound {total_files} valid images to process in {total_batches} batches")
            print("\nStarting batch processing...")

            # Process images in batches
            batch_size = 32
            for batch_num, i in enumerate(range(0, total_files, batch_size), 1):
                batch_paths = image_paths[i:i + batch_size]
                current_batch_size = len(batch_paths)

                try:
                    # Clear previous line and show current batch progress
                    print(f"\033[K\rProcessing batch {batch_num}/{total_batches} "
                          f"({current_batch_size} images)", end="", flush=True)

                    # Process batch with YOLO
                    results = self.model(batch_paths, stream=True)
                    ids, documents, metadatas, embeddings = [], [], [], []

                    # Process each result in the batch
                    for file_path, result in zip(batch_paths, results):
                        try:
                            # Get detections
                            detections = []
                            objects = []
                            if result.boxes:
                                for box in result.boxes:
                                    cls = int(box.cls[0])
                                    conf = float(box.conf[0])
                                    class_name = result.names[cls]
                                    detections.append(
                                        f"{class_name}:{conf:.2f}")
                                    objects.append(class_name)

                            # Create text description
                            text_desc = ", ".join(
                                objects) if objects else "no objects detected"
                            detections_str = "|".join(
                                detections) if detections else "no detections"

                            # Append to batch lists
                            ids.append(file_path)
                            documents.append(text_desc)
                            metadatas.append({
                                "path": file_path,
                                "detections": detections_str,
                                "object_count": len(objects)
                            })
                            embeddings.append(
                                self.create_embeddings(text_desc))
                            processed_files += 1

                        except Exception as e:
                            logger.error(
                                f"Error processing single image {file_path}: {e}")
                            continue

                    # Batch update PostgreSQL with progress indicator
                    if ids:
                        print(f"\033[K\rUploading batch {batch_num}/{total_batches} to PostgreSQL...",
                              end="", flush=True)
                        try:
                            # Start a new transaction
                            self.conn.rollback()  # Reset any failed transaction

                            insert_query = """
                                INSERT INTO media_vectors (path, metadata, embedding)
                                VALUES %s
                                ON CONFLICT (path) DO UPDATE SET
                                    metadata = EXCLUDED.metadata,
                                    embedding = EXCLUDED.embedding
                            """
                            psycopg2.extras.execute_values(
                                self.cursor, insert_query,
                                [(metadata['path'], Json(metadata), embedding)
                                 for metadata, embedding in zip(metadatas, embeddings)]
                            )

                            # Commit the successful inserts
                            self.conn.commit()
                            print(f"\033[K\rCompleted batch {batch_num}/{total_batches} "
                                  f"({processed_files}/{total_files} files processed)")

                        except Exception as e:
                            logger.error(
                                f"Error processing batch {batch_num}: {e}")
                            self.conn.rollback()  # Rollback failed transaction
                            continue

                except Exception as e:
                    logger.error(f"Error processing batch {batch_num}: {e}")
                    continue

            print(
                f"\nIndexing complete. Successfully processed {processed_files}/{total_files} files")
            if skipped_files:
                print(f"Skipped {len(skipped_files)} invalid files")

        except Exception as e:
            logger.error(f"Error during indexing: {e}")
            raise


def main():
    try:
        model_path = os.getenv('MODEL_PATH')
        data_directory = os.getenv('DATA_DIRECTORY')

        if not data_directory:
            logger.error("Please set DATA_DIRECTORY environment variable")
            return

        logger.info(f"Using model: {model_path}")
        logger.info(f"Scanning directory: {data_directory}")

        indexer = MediaIndexer(model_path)
        indexer.index_media(data_directory)

    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()
