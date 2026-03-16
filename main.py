import os
import sys
import logging
import pandas as pd
from tqdm import tqdm

import config
from src.dataset import get_parquet_files, yield_chunks
from src.embedder import Embedder

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def main():
    # Ensure input directory exists
    if not os.path.exists(config.INPUT_DIR):
        logger.error(f"Input directory not found: {config.INPUT_DIR}")
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    logger.info(f"Using input directory: {config.INPUT_DIR}")
    logger.info(f"Using output directory: {config.OUTPUT_DIR}")

    # Gather input files
    input_files = get_parquet_files(config.INPUT_DIR)
    if not input_files:
        logger.warning(f"No .parquet files found in {config.INPUT_DIR}. Exiting.")
        sys.exit(0)

    logger.info(f"Found {len(input_files)} Parquet files to process.")

    # Initialize Embedder (starts multi-process pool)
    try:
        embedder = Embedder()
    except Exception as e:
        logger.error(f"Failed to initialize embedder: {e}")
        sys.exit(1)

    try:
        # We loop through all the chunks lazily via the generator
        chunk_generator = yield_chunks(input_files)
        for output_shard_path, chunk_idx, df_chunk in tqdm(chunk_generator, desc="Processing Chunks"):
            num_rows = len(df_chunk)
            logger.info(f"Encoding chunk {chunk_idx} into {output_shard_path} ({num_rows} rows)...")

            # Extract queries
            queries = df_chunk[config.QUESTION_COL].tolist()

            # Generate embeddings
            embeddings = embedder.encode(queries)

            # Prepare output DataFrame
            output_df = pd.DataFrame()
            output_df[config.QUESTION_COL] = queries

            if config.PRESERVE_ID_COL:
                output_df[config.ID_COL_NAME] = df_chunk[config.ID_COL_NAME].tolist()

            # We must convert the 2D numpy array of embeddings into a list of 1D arrays
            # to store them cleanly in a parquet column
            output_df['embeddings'] = list(embeddings)

            # Save the shard to disk
            output_df.to_parquet(output_shard_path, index=False)
            logger.info(f"Saved {output_shard_path}")

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user. Shutting down gracefully...")
    except Exception as e:
        logger.error(f"An unexpected error occurred during processing: {e}", exc_info=True)
    finally:
        # Ensure the multi-process pool is stopped correctly
        embedder.stop_pool()
        logger.info("Shutdown complete.")

if __name__ == "__main__":
    main()
