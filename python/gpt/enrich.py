import os
import openai
import tqdm


def enrich_event_description(event_description: str, open_ai_key: str | None) -> str:
    """Use GPT3.5-turbo to enrich event description.
    Works in Danish.

    Args:
        event_description (str): Event description.
    Returns:
        str: Enriched event description.
    """
    base_prompt = """
    Efter denne besked følger en beskrivelse af en begivenhed. Tilføj information omkring begivenhedens emner og 
    målgruppe, på formatet der er angivet, ved at udskifte teksten i parenteser med den relevante information.

    EMNER: (tre til seks emner for begivenheden, for eksempel integrationspolitik, danseworkshop, klimaforandringer, etc.)

    MÅLGRUPPE: (tre til seks målgrupper for begivenheden, for eksempel unge, ældre, politikere, sygeplejersker, etc.)
    
    BESKRIVELSE: {event_description}
    {answer_prefix}"""

    answer_prefix = "\n\nEMNER:"

    prompt = base_prompt.format(
        event_description=event_description, answer_prefix=answer_prefix
    )
    if open_ai_key is None:
        open_ai_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = open_ai_key

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return event_description + answer_prefix + response.choices[0].message.content


def enrich_knowledge_base(knowledge_base: dict, open_ai_key: str | None) -> dict:
    """Enrich knowledge base descriptions using GPT to add topic and target audience.
    Args:
        knowledge_base (dict): Knowledge base.
    Returns:
        dict: Enriched knowledge base.
    """
    enriched_knowledge_base = {}
    for k, v in tqdm.tqdm(knowledge_base.items()):
        enriched_knowledge_base[f"enriched_{k}"] = enrich_event_description(
            v, open_ai_key
        )
    return enriched_knowledge_base


def enrich_query(query: str, open_ai_key: str | None) -> str:
    """Enrich query using GPT to add queried topic and target audience, if present. In Danish."""
    base_prompt = """
    Efter denne besked følger forespørgsel om begivenheder, som spørgeren er interesseret i. Du skal IKKE svare på forespørgslen,
    men i stedet TILFØJE relevant information omkring forespørgselen, omkring emner og målgruppe som spørgeren er interesseret i.

    Hvis spørgeren er interesset i begivenheder om bestemte emner, skal disse tilføjes på formatet der er angivet, 
    ved at udskifte teksten i parenteser med den relevante information.

    EMNER SOM SPØRGEREN ER INTERESSERET I: (emner som spørgeren er interesseret i begivenheder omking, for eksempel integrationspolitik, danseworkshop, klimaforandringer, etc.)
    
    Hvis spørgeren ikke eksplicit er interesseret i bestemte emner, skal der stå:     EMNER SOM SPØRGEREN ER INTERESSERET I: N/A.

    Hvis spørgeren er interesset i begivenheder om for bestemte målgrupper, skal disse tilføjes på formatet der er angivet, 
    ved at udskifte teksten i parenteser med den relevante information.

    MÅLGRUPPE SOM SPØRGEREN ER INTERESSERET I: (målgrupper som spørgeren er interesseret i begivenheder fpr, for eksempel unge, ældre, politikere, sygeplejersker, etc.)
    
    Hvis spørgeren ikke eksplicit er interesseret i bestemte målgrupper, skal der stå:     MÅLGRUPPE SOM SPØRGEREN ER INTERESSERET I: N/A.

    Tilføj IKKE yderligere information.

    Forespørgsel: {query}
    {answer_prefix}"""

    answer_prefix = "\n\nEMNER SOM SPØRGEREN ER INTERESSERET I"

    prompt = base_prompt.format(query=query, answer_prefix=answer_prefix)
    if open_ai_key is None:
        open_ai_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = open_ai_key

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    raw_response = answer_prefix + response.choices[0].message.content
    return query + clean_raw_query_enrichment_response(raw_response)


def clean_raw_query_enrichment_response(raw_response: str) -> str:
    """Clean raw response from GPT to add queried topic and target audience, if present. In Danish."""
    lines = raw_response.splitlines()
    response_lines = []
    for line in lines:
        line = line.replace("EMNER SOM SPØRGEREN ER INTERESSERET I", "EMNER")
        line = line.replace("MÅLGRUPPE SOM SPØRGEREN ER INTERESSERET I", "MÅLGRUPPE")
        if "N/A" not in line:
            response_lines.append(line)
    return "\n".join(response_lines)


def enrich_queries(queries: dict, open_ai_key: str | None) -> dict:
    """Enrich queries using GPT to add queried topic and target audience, if present. In Danish."""
    enriched_queries = {}
    for k, v in tqdm.tqdm(queries.items()):
        enriched_queries[f"enriched_{k}"] = enrich_query(v, open_ai_key)
    return enriched_queries
