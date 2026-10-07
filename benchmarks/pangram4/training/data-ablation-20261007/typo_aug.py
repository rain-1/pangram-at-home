"""Typography augmentation for training rows: removes quote style and special spaces as label cues (audit of 2026-10-07).

T2.1's fully AI mirrors use curly quotes almost exclusively while the generic human pool mostly uses straight quotes, and some
Claude/Luna edits put straight apostrophes into curly-quote papers. For each span-supervised training row, one style is drawn
(straight or curly, 50/50, seeded per row) and every quote in the row is rendered in it; both labels get the same style.
Non-breaking and thin spaces become spaces. Every substitution is one character for one character, so region offsets are
unchanged. Document-only rows (stored token labels) are left alone, and a row whose rendered text exceeds the token budget
under the trainer's tokenizer (TOK, set by the trainer) keeps its original text. Selection/calibration windows are never touched.
"""
import random
TOK = None
OPEN_AFTER = set(' \t\n\r([{<—–-/ ')
STRAIGHT = str.maketrans({'“': '"', '”': '"', '„': '"', '‘': "'", '’': "'", '‚': "'", ' ': ' ', ' ': ' ', ' ': ' ', ' ': ' '})


def curly(t):
    out = []
    for i, ch in enumerate(t):
        prev = t[i - 1] if i else ' '
        if ch == '"':
            out.append('“' if prev in OPEN_AFTER else '”')
        elif ch == "'":
            out.append('‘' if prev in OPEN_AFTER else '’')
        else:
            out.append(ch)
    return ''.join(out)


def render(row, seed, limit=510):
    if row.get('supervision') == 'document_only':
        return row
    rng = random.Random(f"{seed}:{row.get('draw_id') or row.get('id')}")
    text = row['text'].translate(STRAIGHT)
    if rng.random() < .5:
        text = curly(text)
    assert len(text) == len(row['text'])
    if text == row['text']:
        return row
    if TOK is not None and len(TOK(text, add_special_tokens=False)['input_ids']) > limit:
        return row
    return {**row, 'text': text}
