import os
import pickle

import pandas as pd
from scipy.spatial.distance import cosine as cosine_distance


def load_knowledge_base(folder_name: str) -> dict:
    """Load knowledge base from a folder.
    Args:
        folder_name (str): Folder name.
    Returns:
        dict: Knowledge base.
    """
    knowledge_base = {}
    for file_name in os.listdir(folder_name):
        if file_name.endswith(".txt"):
            with open(os.path.join(folder_name, file_name)) as f:
                knowledge_base[file_name[:-4]] = f.read()
    return knowledge_base


def save_knowledge_base(knowledge_base: dict, folder_name: str) -> None:
    """Save knowledge base to a folder.
    Args:
        knowledge_base (dict): Knowledge base.
        folder_name (str): Folder name.
    """
    os.makedirs(folder_name, exist_ok=True)
    for k, v in knowledge_base.items():
        with open(os.path.join(folder_name, k + ".txt"), "w") as f:
            f.write(v)


def save_embeddings_to_pickle_file(
    embeddings: dict, folder_name: str, file_name: str
) -> None:
    """Save embeddings to a folder, using pickle.
    Args:
        embeddings (dict): Embeddings.
        folder_name (str): Folder name.
    """
    os.makedirs(folder_name, exist_ok=True)
    if not file_name.endswith(".pkl"):
        file_name += ".pkl"
    with open(os.path.join(folder_name, file_name), "wb") as f:
        pickle.dump(embeddings, f)


def load_embeddings_from_pickle_file(folder_name: str, file_name: str) -> dict:
    """Load embeddings from a folder, using pickle.
    Args:
        folder_name (str): Folder name.
    Returns:
        dict: Embeddings.
    """
    if not file_name.endswith(".pkl"):
        file_name += ".pkl"
    with open(os.path.join(folder_name, file_name), "rb") as f:
        return pickle.load(f)


def cosine_similarities(
    embedded_knowledge_base: dict, embedded_queries: dict
) -> pd.DataFrame:
    """Compute cosine similarities between embedded knowledge base and embedded queries.
    Args:
        embedded_knowledge_base (dict): Embedded knowledge base.
        embedded_queries (dict): Embedded queries.
    Returns:
        pd.DataFrame: Cosine similarities.
    """
    df_sim = pd.DataFrame(
        0,
        index=sorted(embedded_knowledge_base.keys()),
        columns=sorted(embedded_queries.keys()),
    )

    keys_1 = sorted(embedded_knowledge_base.keys())
    keys_2 = sorted(embedded_queries.keys())
    for k1 in keys_1:
        for k2 in keys_2:
            v1 = embedded_knowledge_base[k1]
            v2 = embedded_queries[k2]
            df_sim.loc[k1, k2] = 1 - cosine_distance(v1, v2)
    return df_sim


if __name__ == "__main__":
    knowledge_base = load_knowledge_base("knowledge_base")

    for k, v in knowledge_base.items():
        print(k, v)
        print("\n----\n")
