"""Transparent syllable estimates, not forced-alignment or linguistic ground truth."""
import re
import unicodedata

TOKEN = re.compile(r"[\w\u0900-\u097f]+", re.UNICODE)
EN_EXCEPTIONS = {"people": 2, "every": 2, "being": 2, "fire": 1, "our": 1, "queue": 1}
HI_EXCEPTIONS = {"आपने": 2, "आप": 1, "क्यों": 1, "है": 1}


def syllables(word: str, language: str = "english") -> int:
    word = unicodedata.normalize("NFC", word).lower()
    if word in HI_EXCEPTIONS:
        return HI_EXCEPTIONS[word]
    if re.search(r"[\u0900-\u097f]", word):
        count = 0
        for i, ch in enumerate(word):
            if "\u0904" <= ch <= "\u0914":
                count += 1
            elif "\u0915" <= ch <= "\u0939":
                rest = word[i + 1:]
                if rest.startswith("़"):
                    rest = rest[1:]
                if rest.startswith("्"):
                    continue
                # A matra replaces, rather than adds to, the inherent vowel.
                if not rest and len(word) > 2 and count:
                    continue
                count += 1
        return max(1, count)
    clean = re.sub("[^a-z]", "", word)
    if not clean:
        return 0
    if clean in EN_EXCEPTIONS:
        return EN_EXCEPTIONS[clean]
    if language != "english":
        return max(1, len(re.findall(r"aa|ee|oo|ai|au|ei|ou|[aeiouy]", clean)))
    count = len(re.findall(r"[aeiouy]+", clean))
    if clean.endswith("e") and not clean.endswith(("le", "ye")) and count > 1:
        count -= 1
    return max(1, count)


def cadence(text: str, seconds: float, language: str) -> float:
    return sum(syllables(w, language) for w in TOKEN.findall(text)) / max(seconds, 0.01)


def whisper_options(language: str, hint: str = "") -> dict:
    return {
        "language": {"english": "en", "hindi": "hi", "hinglish": None}[language],
        "word_timestamps": True, "vad_filter": True,
        "initial_prompt": hint[:800] or {
            "english": "Clear English narration.", "hindi": "स्पष्ट हिन्दी में बातचीत।",
            "hinglish": "Hindi aur English code-switching. Keep English technical terms accurate.",
        }[language],
    }
