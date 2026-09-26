"""Focused replay and provenance checks for the CoAuthor span builder."""
import unittest

from build_span_realistic_eval_v1 import apply_events, replay_text_independently, spans_from_labels


def session(events):
    return [{
        "eventName": "system-initialize", "eventSource": "api", "eventNum": 0,
        "currentDoc": "P😀", "currentCursor": 3,
    }, *events]


class ReplayTests(unittest.TestCase):
    def test_utf16_offsets_and_provenance_for_simple_insertions(self):
        events = session([
            {"eventName": "text-insert", "eventSource": "user", "eventNum": 1,
             "currentCursor": 4, "textDelta": {"ops": [{"retain": 3}, {"insert": "x"}]}},
            {"eventName": "text-insert", "eventSource": "api", "eventNum": 2,
             "currentCursor": 6, "textDelta": {"ops": [{"retain": 4}, {"insert": "YZ"}]}},
        ])
        text, labels, _ = apply_events(events)
        self.assertEqual(text, "P😀xYZ")
        self.assertEqual(labels, [None, None, 0, 1, 1])
        self.assertEqual(text, replay_text_independently(events))
        self.assertEqual(spans_from_labels(labels), [
            {"start": 0, "end": 2, "label": -100},
            {"start": 2, "end": 3, "label": 0},
            {"start": 3, "end": 5, "label": 1},
        ])

    def test_user_insert_inside_generated_span_masks_entire_suggestion(self):
        events = session([
            {"eventName": "text-insert", "eventSource": "api", "eventNum": 1,
             "currentCursor": 5, "textDelta": {"ops": [{"retain": 3}, {"insert": "ab"}]}},
            {"eventName": "text-insert", "eventSource": "user", "eventNum": 2,
             "currentCursor": 5, "textDelta": {"ops": [{"retain": 4}, {"insert": "!"}]}},
        ])
        text, labels, edited = apply_events(events)
        self.assertEqual(text, "P😀a!b")
        self.assertEqual(labels, [None] * 5)
        self.assertEqual(edited, 1)
        self.assertEqual(text, replay_text_independently(events))

    def test_replace_ai_text_masks_replacement_and_surviving_origin(self):
        events = session([
            {"eventName": "text-insert", "eventSource": "api", "eventNum": 1,
             "currentCursor": 5, "textDelta": {"ops": [{"retain": 3}, {"insert": "AB"}]}},
            {"eventName": "text-insert", "eventSource": "user", "eventNum": 2,
             "currentCursor": 4, "textDelta": {"ops": [{"retain": 3}, {"delete": 1}, {"insert": "X"}]}},
        ])
        text, labels, edited = apply_events(events)
        self.assertEqual(text, "P😀XB")
        self.assertEqual(labels, [None] * 4)
        self.assertEqual(edited, 1)
        self.assertEqual(text, replay_text_independently(events))

    def test_multichar_user_delta_is_masked_as_paste_ambiguous(self):
        events = session([
            {"eventName": "text-insert", "eventSource": "user", "eventNum": 1,
             "currentCursor": 8, "textDelta": {"ops": [{"retain": 3}, {"insert": "paste"}]}},
        ])
        text, labels, _ = apply_events(events)
        self.assertEqual(text, "P😀paste")
        self.assertEqual(labels, [None] * len(text))
        self.assertEqual(text, replay_text_independently(events))

    def test_replay_refuses_multiple_initializers(self):
        events = session([])
        events.append(events[0].copy())
        with self.assertRaisesRegex(ValueError, "expected one system-initialize"):
            apply_events(events)

    def test_context_propagates_across_following_single_character_events(self):
        events = session([
            {"eventName": "text-insert", "eventSource": "api", "eventNum": 1,
             "currentCursor": 5, "textDelta": {"ops": [{"retain": 3}, {"insert": "ab"}]}},
            {"eventName": "text-insert", "eventSource": "user", "eventNum": 2,
             "currentCursor": 5, "textDelta": {"ops": [{"retain": 4}, {"insert": "!"}]}},
            {"eventName": "text-insert", "eventSource": "user", "eventNum": 3,
             "currentCursor": 6, "textDelta": {"ops": [{"retain": 5}, {"insert": "?"}]}},
        ])
        text, labels, edited = apply_events(events)
        self.assertEqual(text, "P😀a!?b")
        self.assertEqual(labels, [None] * 6)
        self.assertEqual(edited, 2)
        self.assertEqual(text, replay_text_independently(events))


if __name__ == "__main__":
    unittest.main()
