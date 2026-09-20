ANSWER_PROMPT = (
    "Answer the query using only the context from SEC 10-K filings below. "
    "Each passage is prefixed with its source and page; cite the ones you "
    "used. Say so if the context does not answer the question. Do not make "
    "anything up.\n\n"
    "Context:\n{context}\n\n"
    "Query: {question}\n\n"
    "Answer:"
)

HYDE_PROMPT = (
    "Write a short passage from a SEC 10-K filing that would answer the question "
    "below. Invent plausible specifics. The passage is used only as a retrieval "
    "query, so it is never shown to a user and need not be factual.\n\n"
    "Question: {question}\n\n"
    "Passage:"
)

TRIPLE_PROMPT = (
    "Extract the factual relationships stated in the passage below from a SEC "
    "10-K filing. Reply with a JSON list of objects, each holding exactly the "
    'keys "subject", "relation" and "object". Use the wording of the passage. '
    "Reply with an empty list if it states no relationships.\n\n"
    "Passage:\n{text}\n\n"
    "JSON:"
)

__all__ = ["ANSWER_PROMPT", "HYDE_PROMPT", "TRIPLE_PROMPT"]
