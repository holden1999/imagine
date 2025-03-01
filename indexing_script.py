import os
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import chromadb
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
        """Initialize the indexer with YOLO model, SentenceTransformer and ChromaDB"""
        try:
            # Initialize YOLO
            self.model = YOLO(model_path)
            logger.info("YOLO model loaded successfully")

            # Initialize SentenceTransformer
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("SentenceTransformer loaded successfully")

            # Initialize ChromaDB
            self.chroma_client = chromadb.HttpClient()
            logger.info("ChromaDB client initialized")

            # Create or get collection with retry
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.collection = self.chroma_client.get_or_create_collection(
                        name='media_vectors'
                    )
                    logger.info(
                        "ChromaDB collection created/accessed successfully")
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(
                        f"Attempt {attempt + 1} failed, retrying...")
                    time.sleep(1)

        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def create_embeddings(self, text: str) -> list:
        """Create embeddings from text using SentenceTransformer"""
        return self.embedder.encode(text).tolist()

    def index_media(self, directory: str):
        """Index all media files in the directory"""
        logger.info(f"Starting media indexing in directory: {directory}")
        total_files = 0
        processed_files = 0

        # Collect all image paths
        image_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    file_path = os.path.join(root, file)
                    image_paths.append(file_path)
                    total_files += 1

        if not image_paths:
            logger.warning(f"No images found in directory: {directory}")
            return

        # Process images in batches
        batch_size = 32
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            try:
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
                                detections.append(f"{class_name}:{conf:.2f}")
                                objects.append(class_name)

                        # Create text description
                        text_desc = ", ".join(
                            objects) if objects else "no objects detected"

                        # Convert detections list to string
                        detections_str = "|".join(
                            detections) if detections else "no detections"

                        # Append to batch lists
                        ids.append(file_path)
                        documents.append(text_desc)
                        metadatas.append({
                            "path": file_path,
                            "detections": detections_str,  # Store as string instead of list
                            "object_count": len(objects)
                        })
                        embeddings.append(self.create_embeddings(text_desc))

                        processed_files += 1
                        if processed_files % 10 == 0:
                            logger.info(
                                f"Progress: {processed_files}/{total_files}")

                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
                        continue

                # Batch update ChromaDB
                if ids:
                    self.collection.upsert(
                        ids=ids,
                        embeddings=embeddings,
                        documents=documents,
                        metadatas=metadatas
                    )

            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                continue

        logger.info(
            f"Indexing complete. Processed {processed_files}/{total_files} files")


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
