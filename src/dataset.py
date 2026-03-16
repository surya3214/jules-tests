import os
import pyarrow.parquet as pq
import pandas as pd
import logging
from typing import Iterator, Tuple

import config

logger = logging.getLogger(__name__)

from collections import defaultdict

def get_grouped_parquet_files(input_dir: str) -> dict[str, list[str]]:
    """
    Retrieve parquet files and group them by language.
    Requires files to match '{lang}_webfaq.parquet'.
    If config.TARGET_LANGUAGES is set, only includes those languages.
    Returns: { "en": ["file1.parquet", ...], "es": ["file2.parquet", ...] }
    """
    grouped_files = defaultdict(list)

    for root, _, files in os.walk(input_dir):
        for file in sorted(files):
            if not file.endswith('.parquet'):
                continue

            # Expected format: {lang}_webfaq.parquet
            if "_webfaq.parquet" in file:
                lang_prefix = file.replace("_webfaq.parquet", "")

                # Apply TARGET_LANGUAGES filter if set
                if config.TARGET_LANGUAGES is not None and lang_prefix not in config.TARGET_LANGUAGES:
                    continue

                full_path = os.path.join(root, file)
                grouped_files[lang_prefix].append(full_path)
            else:
                logger.warning(f"Skipping {file} because it does not match '{{lang}}_webfaq.parquet'")

    return dict(grouped_files)

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

def yield_language_chunks(lang: str, file_paths: list[str]) -> Iterator[Tuple[str, str, int, pd.DataFrame]]:
    """
    Generator that yields chunks sequentially for a specific language's files.
    """
    for file_path in file_paths:
        try:
            parquet_file = pq.ParquetFile(file_path)
        except Exception as e:
            logger.error(f"[{lang}] Failed to read Parquet file {file_path}: {e}")
            continue

        columns_to_load = [config.QUESTION_COL]
        if config.PRESERVE_ID_COL:
            columns_to_load.append(config.ID_COL_NAME)

        schema_names = parquet_file.schema.names
        missing_cols = [col for col in columns_to_load if col not in schema_names]

        if missing_cols:
            logger.error(f"[{lang}] File {file_path} missing columns: {missing_cols}. Skipping.")
            continue

        chunk_index = 0
        try:
            for batch in parquet_file.iter_batches(batch_size=config.CHUNK_SIZE, columns=columns_to_load):
                output_shard_path = get_output_path(config.OUTPUT_DIR, config.INPUT_DIR, file_path, chunk_index)

                # Resumption logic
                if os.path.exists(output_shard_path):
                    logger.info(f"[{lang}] Resuming: Shard {output_shard_path} already exists. Skipping chunk {chunk_index}.")
                    chunk_index += 1
                    continue

                df_chunk = batch.to_pandas()
                original_len = len(df_chunk)
                df_chunk = df_chunk.dropna(subset=[config.QUESTION_COL])

                if len(df_chunk) < original_len:
                    logger.warning(f"[{lang}] Dropped {original_len - len(df_chunk)} rows with NaN in {config.QUESTION_COL}.")

                if df_chunk.empty:
                    logger.warning(f"[{lang}] Chunk {chunk_index} is empty after dropping NaNs. Skipping.")
                    chunk_index += 1
                    continue

                # We yield exactly what is needed for encoding
                yield lang, output_shard_path, chunk_index, df_chunk
                chunk_index += 1

        except Exception as e:
            logger.error(f"[{lang}] Error reading batches from {file_path}: {e}")
            continue

def yield_interleaved_chunks(grouped_files: dict[str, list[str]]) -> Iterator[Tuple[str, str, int, pd.DataFrame]]:
    """
    Round-robin generator. It takes one chunk from language A, one from B, etc.
    If a language runs out of chunks, it is removed from the rotation,
    but the other languages continue until all data is processed.
    """
    # Create a generator for each language
    generators = {
        lang: yield_language_chunks(lang, files)
        for lang, files in grouped_files.items()
    }

    # We maintain a list of active language keys to cycle through
    active_langs = list(generators.keys())

    while active_langs:
        # Loop over a snapshot of the active languages in this round
        for lang in list(active_langs):
            try:
                # Pull exactly 1 chunk (up to CHUNK_SIZE) for this language
                chunk_data = next(generators[lang])
                yield chunk_data
            except StopIteration:
                # This language generator is exhausted
                logger.info(f"Language '{lang}' has finished processing all its files.")
                active_langs.remove(lang)
