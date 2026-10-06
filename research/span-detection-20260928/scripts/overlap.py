"""Sampled 24-word phrase fingerprints for overlap and leakage audits."""
import hashlib
import re


def norm_words(text):
    return re.findall(r'\w+', text.casefold())


def phrase_fingerprints(text):
    # Keep about 1/16 of 24-word shingles so large corpora fit in memory.
    words = norm_words(text)
    result = set()
    for i in range(max(0, len(words)-23)):
        h = hashlib.blake2b(' '.join(words[i:i+24]).encode(), digest_size=8).digest()
        if h[0] < 16:
            result.add(h)
    return result
