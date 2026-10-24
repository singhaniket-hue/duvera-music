"""One instrumental contract at planning, caching, and provider boundaries."""
import re

MANDATORY_INSTRUMENTAL_SUFFIX = (
    "strictly instrumental only, no vocals, no singing, no voice, no choir, "
    "no vocal chops, no humming, no chanting, no rap, no lyrics, backing track bed only"
)
VOCAL_TERMS = re.compile(r"\b(vocals?|singing|singers?|voices?|choir|lyrics?|humming|chants?|chanting|rap|rapping)\b", re.I)


def instrumental_prompt(prompt: str) -> str:
    prefix = "Instrumental backing music for spoken narration. "
    suffix = ". " + MANDATORY_INSTRUMENTAL_SUFFIX
    prompt = prompt.removeprefix(prefix).removesuffix(suffix)
    # Drop conflicting vocal clauses (including prior negative clauses), then apply one canonical rule.
    clauses = re.split(r"[,;\n]+", prompt)
    safe = [c.strip(" .") for c in clauses if c.strip(" .") and not VOCAL_TERMS.search(c)]
    body = re.sub(r"\bsongs?\b", "instrumental bed", "; ".join(safe), flags=re.I)
    # The official music prompt contract permits 4100 characters. Preserve the constraints when truncating.
    return prefix + body[:4000 - len(prefix) - len(suffix)] + suffix
