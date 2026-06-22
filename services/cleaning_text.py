import re
import textwrap


def clean_text(text):
    # remove weird unicode
    text = re.sub(r'[\uf000-\uf0ff]', '', text)
    # remove long dotted placeholders
    text = re.sub(r'\.{3,}', ' ', text)
    # remove repeated special chars
    text = re.sub(r'[-_]{3,}', ' ', text)
    # fix broken words like "w slr"
    text = re.sub(r'\b[a-z]\s+(?=[a-z])', '', text)
    # normalize spaces BUT preserve newlines
    text = re.sub(r'[ \t]+', ' ', text)
    # remove page numbers like "123"
    text = re.sub(r'\b\d{1,3}\b', ' ', text)
    # fix broken newlines
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def is_noise(text):
    text = text.lower()

    if any(x in text for x in [
        "practice questions",
        "session",
        "ppt",
        "exercise",
        "objective"
    ]):
        return True

    if "?" in text:
        return True

    return False

def get_clean_snippet(text):
    sentences = text.split(".")

    for s in sentences:
        s = s.strip()

        # ignore very short / broken sentences
        if len(s) > 40:
            return textwrap.shorten(s, width=500, placeholder="...")

    # fallback if nothing found
    return textwrap.shorten(text, width=500, placeholder="...")
