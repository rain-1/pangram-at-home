"""Make source-exclusive, title-matched generation prompts for science training."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/mnt/f/pangram-at-home/data/science_articles_v9')
MODELS = ('qwen2_5_3b', 'smollm2_1_7b')


def main():
    rows = [json.loads(line) for line in (ROOT/'train_candidates.jsonl').open()]
    heldout = {json.loads(line)['id'] for name in ('calibration_human.jsonl', 'locked_test_human.jsonl')
               for line in (ROOT/name).open()}
    assert len(rows) == 146 and not heldout.intersection(r['id'] for r in rows)
    outputs = {model: [] for model in MODELS}
    for row in rows:
        model = MODELS[int(hashlib.sha256(row['id'].encode()).hexdigest(), 16) % 2]
        words = min(650, max(450, row['words']))
        prompt = (f'Write an original science feature of about {words} words for a general audience '
                  'on the topic below. Explain the scientific context, evidence, and limitations '
                  'in coherent paragraphs. Write the article itself, with no title, byline, bullet '
                  'list, or introductory note. Do not quote or paraphrase a particular published '
                  f'article.\n\nTopic: {row["title"]}')
        outputs[model].append({'human_id': row['id'], 'split': 'train_ai_candidate',
                               'source': row['source'], 'topic_title': row['title'],
                               'human_text_sha256': row['text_sha256'],
                               'prompt_version': 'science_feature_topic_v2_length_matched',
                               'target_words': words, 'prompt': prompt})
    for model, prompts in outputs.items():
        path = ROOT/f'train_prompts_{model}.jsonl'
        path.write_text(''.join(json.dumps(p, ensure_ascii=False)+'\n' for p in prompts))
        print(model, len(prompts), path)


if __name__ == '__main__':
    main()
