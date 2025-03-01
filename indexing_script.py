import os
from pathlib import Path
from typing import Optional, Dict, Any
import coremltools as ct
import numpy as np
import chromadb
from PIL import Image
import logging
import uuid
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MediaIndexer:
    def __init__(self, model_path: str):
        """Initialize the indexer with CoreML model and ChromaDB"""
        try:
            self.model = ct.models.MLModel(model_path)

            # Get the spec from the MLModel
            spec = self.model.get_spec()

            # Print the input/output description for the MLModel
            # print(spec.description)

            # Get the type of MLModel (NeuralNetwork, SupportVectorRegressor, Pipeline etc)
            print(spec.WhichOneof('Type'))

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

    def process_image(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Process image using CoreML model"""
        try:
            logger.info(f"Processing image: {image_path}")

            # Open and convert image to RGB
            img = Image.open(image_path).convert('RGB')

            # Resize image to 416x416 for YOLOv3
            img = img.resize((416, 416))

            # Make prediction using CoreML model with the correct input shape
            prediction = self.model.predict({
                'image': img
            })
            logger.info(f"Prediction successful: {prediction.keys()}")

            return {
                'file_path': image_path,
                'prediction': prediction,
                'vector': prediction.get('output', [])
            }
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            logger.error(f"Error details: {str(e)}")
            return None

    def index_media(self, directory: str):
        """Index all media files in the directory"""
        logger.info(f"Starting media indexing in directory: {directory}")
        total_files = 0
        processed_files = 0

        for root, _, files in os.walk(directory):
            total_files += len(files)
            for file in files:
                file_path = os.path.join(root, file)

                # Process only images
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    result = self.process_image(file_path)
                    if result:
                        processed_files += 1
                        # Convert prediction vector to string for storage
                        vector_str = ','.join(map(str, result['vector'])) if isinstance(
                            result['vector'], (list, np.ndarray)) else str(result['vector'])

                        # Get tags from prediction if available
                        tags = result.get('prediction', {}).get('labels', [])
                        tags_str = ','.join(tags) if tags else ''

                        # Combine tags and vector for document content
                        document_content = f"tags:{tags_str}|vector:{vector_str}"

                        # Update ChromaDB document
                        self.collection.upsert(
                            ids=[file_path],
                            documents=[document_content],
                            metadatas=[{
                                'type': 'image',
                                'prediction': str(result['prediction']),
                                'file_path': file_path
                            }]
                        )

        logger.info(
            f"Indexing complete. Processed {processed_files} out of {total_files} files")


def main():
    try:
        # Get paths from environment variables
        model_path = os.getenv('MODEL_PATH')
        data_directory = os.getenv('DATA_DIRECTORY')

        if not model_path or not data_directory:
            logger.error(
                "Please set MODEL_PATH and DATA_DIRECTORY environment variables")
            return

        logger.info(f"Using model: {model_path}")
        logger.info(f"Scanning directory: {data_directory}")

        # Initialize indexer with CoreML model
        indexer = MediaIndexer(model_path)

        # Index media files
        indexer.index_media(data_directory)
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()
