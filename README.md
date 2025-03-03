# Image Search and Retrieval System

A Python-based system for indexing and searching images using YOLO object detection, sentence embeddings, and vector similarity search.

## Features

- Object detection using YOLO model
- Text embedding generation using SentenceTransformer
- Vector storage and similarity search with ChromaDB
- Batch processing for efficient image handling
- Environmental variable configuration
- Detailed logging system

## Prerequisites

- Python 3.10 or higher
- ChromaDB server running
- Sufficient storage for image processing and vector storage

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd imagine
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root:
```properties
MODEL_PATH=/path/to/yolo/model.pt
DATA_DIRECTORY=/path/to/image/directory
```

## Usage

### Indexing Images

Run the indexing script to process and store image vectors:

```bash
python indexing_script.py
```

This will:
- Load images from the specified directory
- Detect objects using YOLO
- Generate text embeddings
- Store results in ChromaDB

### Searching Images

Run the retrieval script to search for images:

```bash
python retrieval_script.py
```

Example queries are included in the script:
- "outdoor scene with trees and sky"
- Add more queries in the `queries` list

## Project Structure

```
imagine/
├── .env                    # Environment variables
├── .gitignore             # Git ignore rules
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── indexing_script.py     # Image processing and indexing
└── retrieval_script.py    # Image search and retrieval
```

## Dependencies

- `ultralytics`: YOLO object detection
- `sentence-transformers`: Text embedding generation
- `chromadb`: Vector storage and similarity search
- `python-dotenv`: Environment variable management
- See `requirements.txt` for complete list

## Environment Variables

- `MODEL_PATH`: Path to YOLO model file
- `DATA_DIRECTORY`: Directory containing images to process
- `CHROMADB_COLLECTION`: Name of ChromaDB collection

## Logging

The system provides detailed logging:
- Initialization status
- Processing progress
- Error reporting
- Query results

## Error Handling

- Retry mechanism for ChromaDB operations
- Batch processing error recovery
- Detailed error logging
- Exception handling for file operations

## License

[Your License Here]

## Contributing

[Contributing Guidelines]