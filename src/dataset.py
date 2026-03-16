import os
import pyarrow.parquet as pq
import pandas as pd
import logging
from typing import Iterator, Tuple

import config

logger = logging.getLogger(__name__)

def get_parquet_files(input_dir: str) -> list[str]:
    """
    Retrieve parquet files from the input directory recursively.
    If config.TARGET_LANGUAGES is a list, it only includes files matching
    the '{lang}_webfaq.parquet' format where {lang} is in the list.
    """
    parquet_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.endswith('.parquet'):
                continue

            # If language filtering is enabled, check the filename
            if config.TARGET_LANGUAGES is not None:
                # Expected format: {lang}_webfaq.parquet
                if "_webfaq.parquet" in file:
                    lang_prefix = file.replace("_webfaq.parquet", "")
                    if lang_prefix not in config.TARGET_LANGUAGES:
                        continue
                else:
                    # If it doesn't match the expected naming convention but filtering
                    # is turned on, we should skip it to be safe.
                    continue

            parquet_files.append(os.path.join(root, file))

    return sorted(parquet_files)

def get_output_path(output_dir: str, input_dir: str, file_path: str, chunk_index: int) -> str:
    """Generate the output shard path for a specific file chunk, maintaining folder structure."""
    # Get the relative path of the file from the input directory
    rel_path = os.path.relpath(file_path, input_dir)

    # Create the corresponding directory structure in the output directory
    rel_dir = os.path.dirname(rel_path)
    target_dir = os.path.join(output_dir, rel_dir)
    os.makedirs(target_dir, exist_ok=True)

    base_name = os.path.basename(file_path)
    name_without_ext = os.path.splitext(base_name)[0]
    output_filename = f"{name_without_ext}_shard_{chunk_index:04d}.parquet"
    return os.path.join(target_dir, output_filename)

def yield_chunks(file_paths: list[str]) -> Iterator[Tuple[str, int, pd.DataFrame]]:
    """
    Yields data chunks from a list of parquet files.

    Yields:
        Tuple containing:
            - output_shard_path (str)
            - chunk_index (int)
            - df_chunk (pd.DataFrame)
    """
    for file_path in file_paths:
        logger.info(f"Processing file: {file_path}")

        try:
            parquet_file = pq.ParquetFile(file_path)
        except Exception as e:
            logger.error(f"Failed to read Parquet file {file_path}: {e}")
            continue

        # Determine the columns to load to save memory.
        columns_to_load = [config.QUESTION_COL]
        if config.PRESERVE_ID_COL:
            columns_to_load.append(config.ID_COL_NAME)

        # Verify the necessary columns are present in the file schema
        schema_names = parquet_file.schema.names
        missing_cols = [col for col in columns_to_load if col not in schema_names]

        if missing_cols:
            logger.error(f"File {file_path} is missing required columns: {missing_cols}. Skipping.")
            continue

        chunk_index = 0
        try:
            # Iterate through the parquet file in batches (chunks)
            for batch in parquet_file.iter_batches(batch_size=config.CHUNK_SIZE, columns=columns_to_load):
                output_shard_path = get_output_path(config.OUTPUT_DIR, config.INPUT_DIR, file_path, chunk_index)

                # Resumption logic: if the shard already exists, skip it.
                if os.path.exists(output_shard_path):
                    logger.info(f"Resuming: Shard {output_shard_path} already exists. Skipping chunk {chunk_index}.")
                    chunk_index += 1
                    continue

                df_chunk = batch.to_pandas()

                # Drop rows where the question column is NaN
                original_len = len(df_chunk)
                df_chunk = df_chunk.dropna(subset=[config.QUESTION_COL])
                if len(df_chunk) < original_len:
                    logger.warning(f"Dropped {original_len - len(df_chunk)} rows with NaN in {config.QUESTION_COL} column.")

                if df_chunk.empty:
                    logger.warning(f"Chunk {chunk_index} is empty after dropping NaNs. Skipping.")
                    chunk_index += 1
                    continue

                yield output_shard_path, chunk_index, df_chunk

                chunk_index += 1

        except Exception as e:
            logger.error(f"Error while reading batches from {file_path}: {e}")
            continue
