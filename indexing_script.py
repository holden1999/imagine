import os
from dotenv import load_dotenv
from PIL import Image
import psycopg2
from psycopg2.extras import Json
import logging
from transformers import BlipProcessor, BlipForConditionalGeneration, BartTokenizer, BartModel
from config import config

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MediaIndexer:
    def __init__(self):
        """Initialize the indexer with BLIP processor, model, feature extractor, and PostgreSQL connection"""
        try:
            # Initialize BLIP processor and model
            self.processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-large")
            self.model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-large").to("mps")
            logger.info("BLIP model and processor loaded successfully")

            # Initialize BART tokenizer and model
            self.bart_tokenizer = BartTokenizer.from_pretrained(
                'facebook/bart-base')
            self.bart_model = BartModel.from_pretrained('facebook/bart-base')
            logger.info(
                "facebook/bart-base tokenizer and model loaded for feature extraction")

            # Initialize PostgreSQL connection using config
            db_config = config()
            self.conn = psycopg2.connect(**db_config)
            self.cursor = self.conn.cursor()
            logger.info("PostgreSQL connection initialized")

        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def get_bart_embedding(self, text: str):
        """Extract embedding for the given text using BART model"""
        inputs = self.bart_tokenizer(text, return_tensors="pt")
        outputs = self.bart_model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        embedding = last_hidden_states.mean(dim=1).squeeze().tolist()
        return embedding

    def index_images_with_captions(self, directory: str):
        """Index all images in the directory and store captions in the database"""
        try:
            # Clear terminal
            os.system('clear' if os.name != 'nt' else 'cls')
            logger.info(f"Starting image indexing in directory: {directory}")

            # Collect image paths
            image_files = [
                os.path.join(root, file)
                for root, _, files in os.walk(directory)
                for file in files
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".gif"))
            ]

            if not image_files:
                logger.warning(
                    f"No valid images found in directory: {directory}")
                return

            total_files = len(image_files)
            processed_files = 0

            print(
                f"\nFound {total_files} images to process")
            print("\nStarting processing...")

            # Process each image file
            for img_path in image_files:
                try:
                    # Open and process the image
                    raw_image = Image.open(img_path).convert('RGB')
                    inputs = self.processor(
                        raw_image, return_tensors="pt").to("mps")
                    out = self.model.generate(**inputs)
                    caption = self.processor.decode(
                        out[0], skip_special_tokens=True)

                    # Feature extraction for embedding using BART
                    embedding = self.get_bart_embedding(caption)

                    # Prepare metadata
                    metadata = {"path": img_path, "caption": caption}

                    # Insert or update in PostgreSQL
                    self.cursor.execute(
                        """
                        INSERT INTO media_vectors (path, metadata, embedding)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (path) DO UPDATE SET metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding
                        """,
                        (img_path, Json(metadata), embedding)
                    )
                    self.conn.commit()
                    processed_files += 1
                    print(
                        f"Indexed: {os.path.basename(img_path)} | Caption: {caption}")

                except Exception as e:
                    logger.error(f"Error processing {img_path}: {e}")
                    continue

            print(
                f"\nIndexing complete. Successfully processed {processed_files}/{total_files} files")

        except Exception as e:
            logger.error(f"Error during indexing: {e}")
            raise


def main():
    try:
        data_directory = os.getenv('DATA_DIRECTORY')

        if not data_directory:
            logger.error("Please set DATA_DIRECTORY environment variable")
            return

        logger.info(f"Scanning directory: {data_directory}")

        indexer = MediaIndexer()
        indexer.index_images_with_captions(data_directory)

    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()
