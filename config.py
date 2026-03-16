import os

# ==============================================================================
# CONFIGURATION SETTINGS
# Adjust these variables to match your environment and requirements.
# ==============================================================================

# Paths
INPUT_DIR = os.getenv("INPUT_DIR", "./data/input")      # Folder containing input Parquet files
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./data/output")   # Folder where output Parquet shards will be saved
# To use a local model downloaded on your VM, set this to the local path.
# Example: "./local_qwen_model" instead of "Qwen/Qwen3-Embedding-4B"
MODEL_PATH = os.getenv("MODEL_PATH", "Qwen/Qwen3-Embedding-4B")

# Model Task Configuration
# The instruction prefix required by the Qwen3 embedding model.
TASK_PROMPT = "Given a web search query, retrieve relevant passages that answer the query"

# Data Processing
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))           # Batch size for encoding (applied per GPU process)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 100000))       # Number of rows per output Parquet shard
QUESTION_COL = os.getenv("QUESTION_COL", "question")    # Name of the column containing the text to embed

# Identity Preservation
# Set to True if you want to keep an identifier column to map embeddings back to the original rows.
PRESERVE_ID_COL = os.getenv("PRESERVE_ID_COL", "True").lower() == "true"
ID_COL_NAME = os.getenv("ID_COL_NAME", "id")            # Name of the identifier column in the input files
