import os
import openai

import numpy as np
from tqdm import tqdm


def embed_string(string_to_embed: str, open_ai_key: str | None) -> np.array:
    """Embed a string using GPT-3.
    Args:
        string_to_embed (str): String to embed.
    Returns:
        np.Array: Embedding.
    """
    if open_ai_key is None:
        open_ai_key = os.getenv("OPENAI_API_KEY")
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=string_to_embed,
    )
    return np.array(response.data[0].embedding)


def embed_knowledge_base(knowledge_base: dict, open_ai_key: str | None) -> dict:
    """Embed knowledge base using GPT-3.
    Args:
        knowledge_base (dict): Knowledge base.
    Returns:
        dict: Knowledge base with embeddings.
    """
    embedded_knowledge_base = {}
    for k, v in tqdm(knowledge_base.items()):
        embedded_knowledge_base[k] = embed_string(v, open_ai_key=open_ai_key)
    return embedded_knowledge_base
