"""Encode diagnostics for the current console without changing data or streams."""
import sys


def say(message=""):
    """Print representable text, escaping only characters this stream cannot encode.

    Stream write errors still propagate. Unicode data passed by callers is never
    modified, and UTF-8 consoles retain the original text.
    """
    text = str(message)
    encoding = getattr(sys.stdout, "encoding", None)
    if encoding:
        text = text.encode(encoding, errors="backslashreplace").decode(encoding)
    print(text)
