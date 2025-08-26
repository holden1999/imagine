import pillow_heif  # Add this import
from PIL import Image
from transformers import AutoTokenizer, AutoModel, pipeline, DistilBertTokenizer, DistilBertForQuestionAnswering
from config import config
from dotenv import load_dotenv
from psycopg2.extras import Json
import psycopg2
import logging
import os
import torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
            self.bart_tokenizer = AutoTokenizer.from_pretrained(
                'sentence-transformers/all-mpnet-base-v2')
            self.bart_model = AutoModel.from_pretrained(
                'sentence-transformers/all-mpnet-base-v2')
            logger.info(
                "sentence-transformers/all-mpnet-base-v2 tokenizer and model loaded for feature extraction")
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

    def display_results(self, results, mode="summarize", question=None):
        if not results:
            print("No results found.")
            return

        print("\n=== Search Results ===")
        captions = []
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"File: {os.path.basename(result['path'])}")
            print(f"Caption: {result['caption']}")
            print(f"Distance: {result['distance']:.4f}")
            print("-" * 40)
            captions.append(result['caption'])
            # Preview image using PIL
            try:
                img = Image.open(result['path'])
                img.show(title=os.path.basename(result['path']))
            except Exception as e:
                print(f"Could not preview image: {e}")

        joined_captions = " ".join(captions)
        if mode == "summarize":
            summarizer = pipeline(
                "summarization", model="facebook/bart-large-cnn")
            try:
                summary = summarizer(
                    joined_captions,
                    max_length=60,
                    min_length=15,
                    do_sample=False
                )[0]['summary_text']
                print("\n=== Summary of Captions ===")
                print(summary)
            except Exception as e:
                print(f"Could not summarize captions: {e}")
        elif mode == "qa" and question:
            # Use DistilBERT for Q&A
            qa_tokenizer = DistilBertTokenizer.from_pretrained(
                'distilbert-base-cased-distilled-squad')
            qa_model = DistilBertForQuestionAnswering.from_pretrained(
                'distilbert-base-cased-distilled-squad')
            qa_pipeline = pipeline("question-answering",
                                   model=qa_model, tokenizer=qa_tokenizer)
            try:
                answer = qa_pipeline(
                    question=question, context=joined_captions)
                print("\n=== Q&A Answer ===")
                print(f"Q: {question}")
                print(f"A: {answer['answer']}")
            except Exception as e:
                print(f"Could not answer question: {e}")


def main():
    retriever = ImageRetriever()
    queries = [
        "beach",
    ]
    mode = input("Choose mode ('summarize' or 'qa'): ").strip().lower()
    question = None
    if mode == "qa":
        question = input("Enter your question for Q&A: ").strip()
    for query in queries:
        print(f"\nSearching for: {query}")
        results = retriever.query(query)
        retriever.display_results(results, mode=mode, question=question)


if __name__ == "__main__":
    main()
