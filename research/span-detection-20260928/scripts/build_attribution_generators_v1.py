"""Acquire bounded, prompt-grouped AI response data for later attribution probes.

No API generations occur. Only published response text becomes a model-labeled
example; SynthPrompts is exported separately as prompt-only material.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq


ARENA_REPO = "woog/arena-prose-100-49-models"
ARENA_REV = "66298b561a69c6ed6a9a3c4c79eeedb1f754ba49"
PREF_REPO = "lmarena-ai/arena-human-preference-140k"
PREF_REV = "6322995ab34d7c2693e3f47dd13fa5caa0789a74"
PROMPTS_REPO = "lyraaaa/synthprompts_v2_250k"
PROMPTS_REV = "f286925651e23e7f1d44b22b4f03241dbee9129e"


def sha(data: str | bytes) -> str:
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


def download(repo: str, revision: str, file: str) -> Path:
    return Path(hf_hub_download(repo_id=repo, repo_type="dataset", revision=revision,
                                filename=file))


def normalize_prompt(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+", text, flags=re.UNICODE))


def family(model: str) -> str:
    lower = model.lower()
    for prefix, name in (("anthropic/", "anthropic"), ("claude", "anthropic"),
                         ("openai/", "openai"), ("gpt", "openai"),
                         ("chatgpt", "openai"), ("o3", "openai"), ("o4", "openai"),
                         ("google/", "google"), ("gemini", "google"),
                         ("gemma", "google"), ("qwen/", "qwen"), ("qwen", "qwen"),
                         ("qwq", "qwen"), ("deepseek/", "deepseek"),
                         ("deepseek", "deepseek"), ("mistralai/", "mistral"),
                         ("mistral", "mistral"), ("meta-llama/", "meta"),
                         ("meta/", "meta"), ("llama", "meta"),
                         ("x-ai/", "xai"), ("grok", "xai"),
                         ("command", "cohere"), ("amazon.", "amazon"),
                         ("amazon-", "amazon"), ("minimax", "minimax"),
                         ("moonshotai/", "moonshot"), ("kimi", "moonshot"),
                         ("z-ai/", "zai"), ("xiaomi/", "xiaomi"),
                         ("nvidia/", "nvidia"), ("ibm-granite/", "ibm"),
                         ("upstage/", "upstage"), ("tencent/", "tencent"),
                         ("hunyuan", "tencent"), ("nex-agi/", "nex_agi"),
                         ("inception/", "inception"), ("inclusionai/", "inclusionai"),
                         ("prism-ml/", "prism_ml"), ("dots-studio/", "dots_studio"),
                         ("thinkingmachines/", "thinking_machines"),
                         ("magistral", "mistral")):
        if lower.startswith(prefix):
            return name
    return "other_or_unknown"


def content_text(message: dict) -> str | None:
    content = message.get("content")
    if not isinstance(content, list) or len(content) != 1:
        return None
    item = content[0]
    if item.get("type") != "text" or not isinstance(item.get("text"), str):
        return None
    if item.get("image") or item.get("mimeType"):
        return None
    return item["text"]


def arena_records() -> tuple[list[dict], dict]:
    file = download(ARENA_REPO, ARENA_REV, "data/test-00000-of-00001.parquet")
    rows = pq.read_table(file).to_pylist()
    kept = []
    for row in rows:
        if not row["success"] or not row["mechanically_eligible"]:
            continue
        response, prompt = row["response"], row["prompt"]
        if not response or not prompt or row["language"] != "en":
            continue
        kept.append({"text": response, "prompt_text": prompt, "document_id": "arena_prose:" + row["id"],
                     "source": ARENA_REPO, "source_revision": ARENA_REV,
                     "source_row_id": row["id"], "source_prompt_id": row["prompt_id"],
                     "prompt_id": "arena_prose:" + row["prompt_id"],
                     "prompt_author_status": "historical_platform_user; not_individually_verified",
                     "generator_id": row["model_id"], "canonical_model_reported": row["canonical_model"],
                     "request_model": row["request_model"], "response_model": row["response_model"],
                     "provider": row["provider"], "generator_family": family(row["model_id"]),
                     "prompt_category": row["prompt_category"], "language": row["language"],
                     "generation_params": {"reasoning_setting": row["reasoning_setting"],
                                           "max_tokens": row["max_tokens"],
                                           "temperature": row["temperature"],
                                           "finish_reason": row["finish_reason"]},
                     "label_basis": "response_model_and_route_metadata_reported_by_corpus",
                     "rights_status": "mixed_source_and_model_output_terms; local_research_only",
                     "prose_eligibility": "single_reviewer_open_ended_prompt; mechanical_response_gate",
                     "response_words": row["response_words"],
                     "response_sha256": sha(response), "prompt_sha256": sha(prompt),
                     "source_response_sha256": row["response_sha256"]})
    return kept, {"path": str(file), "sha256": sha(file.read_bytes()), "source_rows": len(rows),
                  "retained": len(kept)}


def preference_records() -> tuple[list[dict], dict]:
    file = download(PREF_REPO, PREF_REV, "data/train-00000-of-00007.parquet")
    rows = pq.read_table(file).to_pylist()
    candidates = []
    exclusions = Counter()
    for row in rows:
        if row["language"] != "en" or row["is_code"]:
            exclusions["non_english_or_code"] += 1
            continue
        if row["category_tag"]["math_v0.1"]["math"]:
            exclusions["math_tag"] += 1
            continue
        a, b = row["conversation_a"], row["conversation_b"]
        if len(a) != 2 or len(b) != 2 or any(x.get("role") != role for conv in (a, b)
                                            for x, role in zip(conv, ("user", "assistant"))):
            exclusions["multi_turn_or_role"] += 1
            continue
        prompts = (content_text(a[0]), content_text(b[0]))
        responses = (content_text(a[1]), content_text(b[1]))
        if None in prompts or None in responses or prompts[0] != prompts[1]:
            exclusions["nontext_or_prompt_mismatch"] += 1
            continue
        prompt = prompts[0]
        if not 8 <= len(prompt.split()) <= 600 or any(len(text.split()) < 80 for text in responses):
            exclusions["length_gate"] += 1
            continue
        candidates.append((row, prompt, responses))
    kept = []
    for row, prompt, responses in candidates:
        for side, text in zip(("a", "b"), responses):
            model = row[f"model_{side}"]
            kept.append({"text": text, "prompt_text": prompt,
                         "document_id": f"lmarena_preference:{row['id']}:{side}",
                         "source": PREF_REPO, "source_revision": PREF_REV,
                         "source_row_id": row["id"],
                         "source_prompt_id": row["evaluation_session_id"] + ":" + row["id"],
                         "prompt_id": "lmarena_preference:" + sha(prompt)[:20],
                         "prompt_author_status": "arena_platform_user; not_individually_verified",
                         "generator_id": model, "canonical_model_reported": None,
                         "request_model": None, "response_model": None, "provider": None,
                         "generator_family": family(model),
                         "prompt_category": ("creative_writing" if row["category_tag"]["creative_writing_v0.1"]["creative_writing"]
                                             else "general_open_ended_candidate"),
                         "language": "en", "generation_params": None,
                         "label_basis": "published_model_a_or_b; backend_version_and_route_unverified",
                         "rights_status": "cc_by_4_dataset; model_output_terms_not_individually_audited",
                         "prose_eligibility": "one_turn_english_noncode_nonmath_length_heuristic; not_human_reviewed",
                         "response_words": len(text.split()), "response_sha256": sha(text),
                         "prompt_sha256": sha(prompt), "timestamp": str(row["timestamp"])})
    return kept, {"path": str(file), "sha256": sha(file.read_bytes()),
                  "source_rows": len(rows), "eligible_pairs": len(candidates),
                  "retained": len(kept), "exclusions": dict(exclusions)}


class DSU:
    def __init__(self, n: int): self.parent = list(range(n))
    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b: self.parent[max(a, b)] = min(a, b)


def prompt_groups(records: list[dict]) -> tuple[dict[str, str], dict]:
    """Group exact and high-overlap five-word-shingle prompt variants."""
    originals = sorted({row["prompt_text"] for row in records}, key=sha)
    norm = [normalize_prompt(x) for x in originals]
    tokens = [x.split() for x in norm]
    shingles = [set(zip(*(words[i:] for i in range(5)))) if len(words) >= 5 else set()
                for words in tokens]
    dsu = DSU(len(originals))
    exact = {}
    index: dict[tuple, list[int]] = defaultdict(list)
    for i, (text, grams) in enumerate(zip(norm, shingles)):
        if text in exact: dsu.union(i, exact[text])
        else: exact[text] = i
        votes = Counter(j for gram in grams for j in index[gram])
        for j, common in votes.items():
            if common / max(len(grams), len(shingles[j])) >= .85:
                dsu.union(i, j)
        for gram in grams:
            index[gram].append(i)
    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(len(originals)):
        clusters[dsu.find(i)].append(i)
    group = {}
    for members in clusters.values():
        gid = "prompt:" + min(sha(norm[i]) for i in members)[:20]
        for i in members: group[originals[i]] = gid
    return group, {"unique_prompt_strings": len(originals), "prompt_groups": len(clusters),
                   "near_duplicate_merges": len(originals) - len(clusters),
                   "max_cluster_prompts": max(map(len, clusters.values()))}


def assign_splits(records: list[dict], groups: dict[str, str]) -> dict[str, str]:
    by_origin: dict[str, set[str]] = defaultdict(set)
    for row in records:
        origin = "arena_prose" if row["source"] == ARENA_REPO else "lmarena_preference"
        by_origin[origin].add(groups[row["prompt_text"]])
    arena = sorted(by_origin["arena_prose"], key=lambda x: sha("arena-split:" + x))
    preference = sorted(by_origin["lmarena_preference"] - set(arena),
                        key=lambda x: sha("preference-split:" + x))
    result = {}
    for order in (arena, preference):
        n = len(order)
        for i, gid in enumerate(order):
            result[gid] = "train" if i < int(.7*n) else "calibration" if i < int(.85*n) else "test"
    return result


def prompt_only_sample(limit: int) -> tuple[list[dict], dict]:
    file = download(PROMPTS_REPO, PROMPTS_REV, "prompts_v2.jsonl")
    candidates = []
    with file.open() as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            prompt = row["prompt"]
            if row["language"] != "english" or not 15 <= len(prompt.split()) <= 450:
                continue
            if row["mode"] == "task" and ("writing" not in row["topic"].lower() and
                                           "essay" not in row["topic"].lower() and
                                           "creative" not in row["topic"].lower()):
                continue
            candidates.append((sha(f"prompt-only-v1:{index}:{prompt}"), index, row))
    selected = sorted(candidates)[:limit]
    output = []
    for _, index, row in selected:
        prompt = row["prompt"]
        output.append({"prompt_id": "synthprompts:" + sha(prompt)[:20], "prompt_text": prompt,
                       "source_row_index": index, "source": PROMPTS_REPO,
                       "source_revision": PROMPTS_REV, "prompt_writer_reported": "google/gemma-4-31b-it",
                       "response_generator": None, "response_text": None,
                       "topic": row["topic"], "mode": row["mode"], "style": row["style"],
                       "length_axis": row["length"], "complexity": row["complexity"],
                       "task_spec_level": row["task_spec_level"], "language": row["language"],
                       "prompt_sha256": sha(prompt), "rights_status": "MIT_dataset_card"})
    return output, {"path": str(file), "sha256": sha(file.read_bytes()),
                    "eligible_prompt_candidates": len(candidates), "selected": len(output)}


def write_jsonl(path: Path, rows: list[dict]) -> dict:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    return {"path": path.name, "rows": len(rows), "sha256": sha(path.read_bytes()),
            "models": dict(Counter(row["generator_id"] for row in rows)),
            "families": dict(Counter(row["generator_family"] for row in rows)),
            "sources": dict(Counter(row["source"] for row in rows)),
            "prompt_groups": len({row["group_id"] for row in rows}),
            "categories": dict(Counter(row["prompt_category"] for row in rows))}


def source_view(records: list[dict], source: str) -> dict:
    subset = [row for row in records if row["source"] == source]
    by_split = {}
    for split in ("train", "calibration", "test"):
        rows = [row for row in subset if row["split"] == split]
        lengths = sorted(row["response_length_words"] for row in rows)
        by_split[split] = {"response_rows": len(rows),
                           "prompt_groups": len({row["group_id"] for row in rows}),
                           "exact_published_model_labels": dict(Counter(row["generator_id"] for row in rows)),
                           "family_taxonomy_counts": dict(Counter(row["generator_family"] for row in rows)),
                           "prompt_categories": dict(Counter(row["prompt_category"] for row in rows)),
                           "response_words_median": lengths[len(lengths)//2] if lengths else None,
                           "self_identification_surface_flags": sum(row["self_identification_surface_flag"] for row in rows)}
    counts = {split: Counter(row["generator_id"] for row in subset if row["split"] == split)
              for split in ("train", "calibration", "test")}
    candidate = [model for model in sorted(counts["train"])
                 if counts["train"][model] >= 60 and counts["calibration"][model] >= 10
                 and counts["test"][model] >= 10]
    return {"source": source, "source_revision": ARENA_REV if source == ARENA_REPO else PREF_REV,
            "use": "within-source exact published-model probe; prompt-group split",
            "filter_unified_records_where_source_equals": source,
            "label_warning": ("Reported route and canonical model retained; not independently verified weights"
                              if source == ARENA_REPO else
                              "Published model_a/model_b only; exact backend revision and route unknown"),
            "splits": by_split, "candidate_exact_model_labels": candidate}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/mnt/f/pangram-at-home"))
    parser.add_argument("--output-folder", default="attribution_generators_v1")
    parser.add_argument("--prompt-only-folder", default="attribution_prompt_only_v1")
    parser.add_argument("--prompt-only-limit", type=int, default=2000)
    args = parser.parse_args()
    output = args.root / "data" / args.output_folder
    prompt_output = args.root / "data" / args.prompt_only_folder
    if output.exists() or prompt_output.exists():
        raise SystemExit(f"Refusing to overwrite {output} or {prompt_output}")
    arena, arena_info = arena_records()
    preference, preference_info = preference_records()
    records = arena + preference
    groups, cluster_info = prompt_groups(records)
    splits = assign_splits(records, groups)
    for row in records:
        row["group_id"] = groups[row["prompt_text"]]
        row["split"] = splits[row["group_id"]]
        row["response_length_words"] = len(row["text"].split())
        row["clean_sha256"] = row["response_sha256"]  # No response rewriting or cleaning.
        row["self_identification_surface_flag"] = bool(re.search(
            r"\b(?:as an? (?:ai|language model)|i am (?:chatgpt|claude|gemini|gpt|an? ai))\b",
            row["text"][:2000], re.IGNORECASE))
    # If an identical response appeared in more than one split, exclude it from
    # all but the first split to preserve exact-text separation.
    split_order = {"train": 0, "calibration": 1, "test": 2}
    records.sort(key=lambda row: (split_order[row["split"]], sha(row["document_id"])))
    seen_responses = set()
    unique = []
    duplicate_responses = 0
    for row in records:
        if row["response_sha256"] in seen_responses:
            duplicate_responses += 1
            continue
        seen_responses.add(row["response_sha256"])
        unique.append(row)
    prompt_only, prompt_info = prompt_only_sample(args.prompt_only_limit)
    output.mkdir(parents=True)
    manifest = {"role": "later frozen-backbone AI generator attribution research; no model trained",
                "response_text_only": True, "prompt_text_separate": True,
                "family_labels_are_prefix_taxonomy_not_provider_proof": True,
                "generator_id_is_published_label_not_guaranteed_weight_revision": True,
                "source_revisions": {ARENA_REPO: ARENA_REV, PREF_REPO: PREF_REV,
                                     PROMPTS_REPO: PROMPTS_REV},
                "sources": {"arena_prose": arena_info, "lmarena_preference": preference_info,
                            "synthprompts_prompt_only": prompt_info},
                "prompt_grouping": cluster_info, "duplicate_response_rows_removed": duplicate_responses,
                "splits": {}}
    parent = args.root / "data/diverse_pyramid_v1"
    if all((parent / f"{split}_full.parquet").exists() for split in ("train", "val", "test")):
        detector_hashes = {row["text_sha256"] for split in ("train", "val", "test")
                           for row in pq.read_table(parent / f"{split}_full.parquet",
                                                    columns=["text_sha256"]).to_pylist()}
        response_hashes = {sha(" ".join(row["text"].casefold().split())) for row in unique}
        manifest["detector_parent_exact_normalized_overlap"] = len(detector_hashes & response_hashes)
        manifest["detector_parent_overlap_limit"] = "Exact normalized text only; no semantic or near-duplicate audit"
    for split in ("train", "calibration", "test"):
        rows = [row for row in unique if row["split"] == split]
        manifest["splits"][split] = write_jsonl(output / f"{split}.jsonl", rows)
        print(split, len(rows), "groups", manifest["splits"][split]["prompt_groups"], flush=True)
    prompt_output.mkdir(parents=True)
    file = prompt_output / "prompts.jsonl"
    file.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prompt_only))
    prompt_manifest = {"role": "future generation prompt pool only; no responses included",
                       "path": file.name, "rows": len(prompt_only),
                       "sha256": sha(file.read_bytes()), "source": PROMPTS_REPO,
                       "source_revision": PROMPTS_REV,
                       "source_file_sha256": prompt_info["sha256"],
                       "prompt_writer_reported": "google/gemma-4-31b-it",
                       "warning": "Model-generated USER PROMPTS, not model answers or human examples"}
    (prompt_output / "manifest.json").write_text(json.dumps(prompt_manifest, indent=2) + "\n")
    manifest["prompt_only_separate_folder"] = str(prompt_output)
    counts = {split: Counter(row["generator_id"] for row in unique if row["split"] == split)
              for split in ("train", "calibration", "test")}
    manifest["candidate_exact_model_labels"] = [model for model in sorted(counts["train"])
                                                 if counts["train"][model] >= 60
                                                 and counts["calibration"][model] >= 10
                                                 and counts["test"][model] >= 10]
    manifest["source_views"] = {}
    for source, name in ((ARENA_REPO, "arena_prose"), (PREF_REPO, "lmarena_preference")):
        view = source_view(unique, source)
        path = output / f"{name}_view_manifest.json"
        path.write_text(json.dumps(view, indent=2) + "\n")
        manifest["source_views"][name] = {"path": path.name, "sha256": sha(path.read_bytes()),
                                          "candidate_exact_model_labels": len(view["candidate_exact_model_labels"])}
    source_models = {source: {row["generator_id"] for row in unique if row["source"] == source}
                     for source in (ARENA_REPO, PREF_REPO)}
    source_families = {source: {row["generator_family"] for row in unique if row["source"] == source}
                       for source in (ARENA_REPO, PREF_REPO)}
    manifest["cross_source"] = {
        "exact_published_model_id_intersection": len(source_models[ARENA_REPO] & source_models[PREF_REPO]),
        "shared_inferred_family_labels": sorted(source_families[ARENA_REPO] & source_families[PREF_REPO]),
        "identity_warning": "Zero shared exact strings does not establish that underlying models differ; no alias mapping verified. A unified exact-model head would be confounded by corpus source."
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
