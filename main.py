from python.gpt.embed import embed_knowledge_base
from python.util import (
    cosine_similarities,
    load_embeddings_from_pickle_file,
    load_knowledge_base,
    save_embeddings_to_pickle_file,
    save_knowledge_base,
)
from python.gpt.enrich import enrich_knowledge_base, enrich_queries

import os

if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")

    ENRICH_EVENTS = False
    ENRICH_QUERIES = False
    EMBED_EVENTS = True
    EMBED_QUERIES = True
    DO_ANALYSIS = True

    if ENRICH_EVENTS:
        knowledge_base = load_knowledge_base("knowledge_base")
        enriched_knowledge_base = enrich_knowledge_base(knowledge_base, api_key)
        save_knowledge_base(enriched_knowledge_base, "knowledge_base/enriched")

    if ENRICH_QUERIES:
        queries = load_knowledge_base("queries")
        enriched_queries = enrich_queries(queries, api_key)
        save_knowledge_base(enriched_queries, "queries/enriched")

    if EMBED_EVENTS:
        knowledge_base = load_knowledge_base(
            "knowledge_base/enriched"
        ) | load_knowledge_base("knowledge_base")
        embedded_knowledge_base = embed_knowledge_base(knowledge_base, api_key)
        save_embeddings_to_pickle_file(
            embedded_knowledge_base,
            "knowledge_base/enriched",
            "embedded_enriched_knowledge_base.pkl",
        )

    if EMBED_QUERIES:
        queries = load_knowledge_base("queries/enriched") | load_knowledge_base(
            "queries"
        )
        embedded_queries = embed_knowledge_base(queries, api_key)
        save_embeddings_to_pickle_file(
            embedded_queries,
            "queries/enriched",
            "embedded_enriched_queries.pkl",
        )

    if DO_ANALYSIS:
        embedded_knowledge_base = load_embeddings_from_pickle_file(
            "knowledge_base/enriched", "embedded_enriched_knowledge_base.pkl"
        )
        embedded_queries = load_embeddings_from_pickle_file(
            "queries/enriched", "embedded_enriched_queries.pkl"
        )
        df_sim = cosine_similarities(embedded_knowledge_base, embedded_queries)
        df_sim.to_csv("similarity.csv")
