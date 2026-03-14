import re
def generate_from_context(context: str, question: str) -> str:
    if not context:
        return "No domain information available to answer the question."
    sentences = [s.strip() for s in re.split(r"[.!?]", context) if s.strip()]
    qwords = [w.lower() for w in re.findall(r"\\w+", question)]
    for s in sentences:
        if any(q in s.lower() for q in qwords):
            return s
    return sentences[0] if sentences else "Unable to extract answer from domain."
