import logging
from typing import List, Any
import numpy as np
from sentence_transformers import SentenceTransformer

import config

logger = logging.getLogger(__name__)

class Embedder:
    def __init__(self, model_path: str = config.MODEL_PATH):
        logger.info(f"Loading SentenceTransformer model from {model_path}...")
        try:
            self.model = SentenceTransformer(model_path)
            # Start multi-process pool for multi-GPU encoding
            logger.info("Starting multi-process pool for multi-GPU execution...")
            self.pool = self.model.start_multi_process_pool()
        except Exception as e:
            logger.error(f"Failed to load model or start multi-process pool: {e}")
            raise

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        """Format the query with the task instruction as required by Qwen3."""
        return f'Instruct: {task_description}\nQuery:{query}'

    def encode(self, queries: List[str]) -> np.ndarray:
        """
        Encode a list of queries using the multi-process pool.
        Note: The Qwen3 embedding model on HF recommends prompt formatting for queries.
        Alternatively, if SentenceTransformers configuration includes the prompt,
        we can use `prompt_name="query"`. We handle the prompt formatting manually
        here to be explicit with `TASK_PROMPT`.
        """
        # Format queries with the specific Qwen3 prompt
        formatted_queries = [
            self.get_detailed_instruct(config.TASK_PROMPT, q)
            for q in queries
        ]

        # Encode using the multi-process pool
        embeddings = self.model.encode_multi_process(
            formatted_queries,
            self.pool,
            batch_size=config.BATCH_SIZE
        )
        return embeddings

    def stop_pool(self):
        """Stop the multi-process pool."""
        if hasattr(self, 'pool') and self.pool is not None:
            logger.info("Stopping multi-process pool...")
            self.model.stop_multi_process_pool(self.pool)
