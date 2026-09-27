"""Prepare topic-only AI mirrors for archived magazine training candidates."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/mnt/f/pangram-at-home/data/smithsonian_archive_v10')
MODELS = ('qwen2_5_3b', 'smollm2_1_7b')


def main():
    train = [json.loads(line) for line in (ROOT/'train_candidates.jsonl').open()]
    test = {json.loads(line)['id'] for line in (ROOT/'locked_test_human.jsonl').open()}
    assert len(train) == 79 and not test.intersection(r['id'] for r in train)
    outputs = {model: [] for model in MODELS}
    for row in train:
        model = MODELS[int(hashlib.sha256(row['id'].encode()).hexdigest(), 16) % 2]
        target = min(650, max(450, row['words']))
        prompt = (f'Write an original magazine feature of about {target} words for a general '
                  'audience on the topic below. Use coherent paragraphs, concrete details, and '
                  'appropriate context. Write only the article body: no title, byline, bullet '
                  'list, or introductory note. Do not quote or paraphrase any particular '
                  f'published article.\n\nTopic: {row["title"]}')
        outputs[model].append({'human_id': row['id'], 'split': 'magazine_train_ai_candidate',
                               'source': row['source'], 'topic_title': row['title'],
                               'human_text_sha256': row['text_sha256'],
                               'prompt_version': 'magazine_feature_topic_v1',
                               'target_words': target, 'prompt': prompt})
    for model, prompts in outputs.items():
        path = ROOT/f'train_prompts_{model}.jsonl'
        path.write_text(''.join(json.dumps(p, ensure_ascii=False)+'\n' for p in prompts))
        print(model, len(prompts), path)


if __name__ == '__main__':
    main()
