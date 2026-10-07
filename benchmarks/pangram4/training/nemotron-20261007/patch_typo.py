"""Add --typo-aug (typo_aug.py) to a sweep trainer in place: python patch_typo.py train_sweep.py. Off by default; idempotent."""
import re, sys
p = sys.argv[1]; s = open(p).read()
if 'typo_aug' in s:
    sys.exit(print('already patched'))
s = s.replace('SHORT_SPAN_OVERSAMPLE = False', 'import typo_aug\nTYPO_AUG = False  # set by --typo-aug\nSHORT_SPAN_OVERSAMPLE = False', 1)
anchor = "    random.Random(seed * 1000 + sum(map(ord, key))).shuffle(short)\n    return short\n"
assert s.count(anchor) == 1
s = s.replace(anchor, "    if TYPO_AUG:\n        short = [typo_aug.render(r, seed) for r in short]\n" + anchor, 1)
s = re.sub(r"(\n    global SHORT_SPAN_OVERSAMPLE\n    SHORT_SPAN_OVERSAMPLE = a\.short_span_oversample\n)",
           r"\1    global TYPO_AUG\n    TYPO_AUG = a.typo_aug\n", s, count=1)
assert 'TYPO_AUG = a.typo_aug' in s
s = s.replace("'short_span_oversample': a.short_span_oversample,", "'short_span_oversample': a.short_span_oversample, 'typo_aug': a.typo_aug,", 1)
m = re.search(r"\n(\s+)(\S.*?)tok = AutoTokenizer\.from_pretrained\(asset, local_files_only=True\)\n", s)
s = s[:m.end()] + m.group(1) + 'typo_aug.TOK = tok\n' + s[m.end():]
s = s.replace("    p.add_argument('--short-span-oversample', action='store_true')",
              "    p.add_argument('--short-span-oversample', action='store_true')\n    p.add_argument('--typo-aug', action='store_true', help='per-row quote-style/space randomisation (typo_aug.py)')", 1)
assert "'--typo-aug'" in s and "'typo_aug': a.typo_aug" in s
open(p, 'w').write(s); print('patched', p)
