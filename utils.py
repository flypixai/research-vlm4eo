import re

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


def split_classes(s: str | None) -> set[str]:
    s = s or ""
    return {cls.strip().lower() for cls in s.split(",") if cls.strip()}


def normalize(token: str) -> str:
    tok = token.lower().strip()
    tok = re.sub(r"[^a-z0-9 ]", "", tok)
    return lemmatizer.lemmatize(tok)


def preprocess_classes(raw: str) -> set[str]:
    classes = split_classes(raw)
    return {normalize(c) for c in classes}
