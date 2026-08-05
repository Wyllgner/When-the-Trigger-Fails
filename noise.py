"""MR2 - deterministic surface noise (typos, doubled letters and spaces).

The same (text, seed) always yields the same output, so the frozen experimental
input in variations.jsonl is reproducible.
"""
import random


def _protected(text):
    """Character indexes that must not be corrupted: tokens carrying essential
    arguments (emails, numbers, dates)."""
    prot = set()
    pos = 0
    for tok in text.split(" "):
        if "@" in tok or any(c.isdigit() for c in tok):
            for k in range(len(tok)):
                prot.add(pos + k)
        pos += len(tok) + 1
    return prot


def add_noise(text, seed=42):
    rng = random.Random(seed)
    prot = _protected(text)
    chars = list(text)

    def free_alpha(i):
        return chars[i].isalpha() and i not in prot

    # 1) duplication typo: double 1-2 random letters
    n_dups = rng.randint(1, 2)
    for _ in range(n_dups):
        idxs = [i for i in range(len(chars)) if free_alpha(i)]
        if not idxs:
            break
        i = rng.choice(idxs)
        chars.insert(i, chars[i])
        prot = {p + 1 if p >= i else p for p in prot}

    # 2) transposition typo: swap one pair of adjacent letters
    idxs = [i for i in range(len(chars) - 1)
            if free_alpha(i) and free_alpha(i + 1)]
    if idxs:
        i = rng.choice(idxs)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]

    text = "".join(chars)

    # 3) double spaces in part of the occurrences
    if rng.random() < 0.7:
        out, double = [], False
        for c in text:
            out.append(c)
            if c == " " and not double and rng.random() < 0.5:
                out.append(" ")
                double = True
            elif c != " ":
                double = False
        text = "".join(out)

    return text


if __name__ == "__main__":
    exemplos = [
        "Send an email to ana@x.com saying the meeting is at 3pm.",
        "What's the weather like in Recife right now?",
    ]
    for s in exemplos:
        print(repr(add_noise(s, seed=42)))
