def split_classes(s: str) -> set[str]:
    return {cls.strip().lower() for cls in s.split(",") if cls.strip()}
