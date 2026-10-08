Hi woog, three things for moe-mixA-full / mixB, from rain1's side. Everything below is on the integration branch (rain-1/pangram-at-home, integrate/woog-workbench-plus-span) and on the Space.

**1. Quote-style shortcut in mixA (stage2-epoch0, per 10k chars of labelled text)**

| | AI | human | ratio |
|---|---:|---:|---:|
| curly " | 12.6 | 3.1 | 4.0x |
| curly ' | 8.0 | 2.7 | 3.0x |
| straight " | 0.7 | 2.7 | 0.29x |
| straight ' | 1.1 | 3.1 | 0.36x |

Mostly the Luna mirrors (curly only, straight 0.0/10k) against the generic human pool (straight quotes, hard wraps, double spaces). Claude/Luna edits also put straight apostrophes into curly-quote papers (6.5% of T2.1 mixed rows). Your sweep-eval rows don't have that cue, so the in-domain numbers aren't inflated by it. But real chatbot output is mostly straight quotes, so "curly = AI" can cost recall and flag curly-quoted human text.

Fix: `typo_aug.py` + `patch_typo.py` add `--typo-aug` to your train_sweep.py. Per row it renders every quote as all-straight or all-curly (50/50, seeded), the same for both labels. NBSP/thin spaces become spaces. Every substitution is one character for one, so offsets are untouched. Document-only rows are skipped, and a row that would exceed 510 tokens keeps its text.
- Files: `/data/workspace/pangram-ablation-20261007/mix-inputs/`
- Audit script: `research/data-mix-20261006/scripts/format_audit_t21.py PREPARED_DIR`
- The 4B A/B (T2.1+A2 with and without --typo-aug, 2 seeds) is running now on GPUs 6-7. Results around 11:00 UTC.

**2. rain1's heterogeneous-ai-spans helps on top of T2.1** (4B, T2.1 recipe, 3 seeds per arm, your full eval battery):

| | all | small | paper_v3 | standalone | t21-heldout | heldout-writers |
|---|---:|---:|---:|---:|---:|---:|
| T2.1 (your 3 seeds) | 0.784 | 0.449 | 0.760 | 0.793 | – | – |
| T2.1 + A2 (layout-neutral hetero) | 0.793 | 0.491 | 0.773 | 0.799 | 0.762 | 0.633 |

The cost: untouched hetero test controls are flagged 1.4% vs 0.4%. Held-out FP rates don't change. Human controls alone, or single sources, hurt held-out writers.
- A2 adds about 2,430 windows per stage-2 shard (Gutenberg, JMLR, Standard Ebooks mixed + JMLR controls; no Beige Book / WikiText).
- Builder: `setup_mix.py --spec specs/A2-neutral.json --additions additions-hetero-neutral.jsonl.gz --drops leakage-drop-ids.json --extra-held <your held-out score rows>`, same folder.
- Results: `benchmarks/pangram4/training/data-ablation-20261007/README.md`

mixA has none of it. Worth adding to mixB?

**3. Mix composition matters a lot on rain1's benchmark-v3.** A Qwen MoE on T2.1+A2 (20% length) beat the prepared-v2 MoE on your eval: small 0.49 vs about 0.25. But it lost a lot of AI recall on v3: Beemo AI 95.5 → 60.0, controlled replacement 69.9 → 32.5. In exchange, human false alarms fell 4-10x. Our guess is the cut in mirrors and generic human (6,000 → 1,400 / 1,700 per shard). mixA restores those, so it may get both.

We're scoring your final-moe (`/tmp/final-20261007/runs-moe`, loaded with your vendor + peft 0.21.2) on benchmark-v3 now on GPU 4, done around 09:55 UTC, so you'll have a v3 comparison.

**GPU use right now (rain1):** GPUs 5-7 run 4B runs (about 12 GB each) until about 11:00 UTC. GPU 4 is scoring until about 09:55 UTC. GPUs 0-1 are left free for you. Last night (about 18:25 UTC) score_moe.py took 6 GPUs while our runs were queued and they crashed out of memory. Could we agree a way to claim GPUs, e.g. a `/tmp/GPU-CLAIMS` file?
