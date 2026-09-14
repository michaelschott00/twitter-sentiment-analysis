import json
import os
import tempfile
import unittest

import pandas as pd

from twitter.baselines.llm_baseline import (
    BATCH_ENDPOINT,
    build_batch_request_line,
    build_chat_body,
    collect_batch_predictions,
    custom_id_for_row,
    extract_content_from_result,
    load_manifest,
    save_manifest,
    write_batch_input_file,
)


def _make_chat_result(custom_id, content, status_code=200):
    return {
        "id": f"batch_req_{custom_id}",
        "custom_id": custom_id,
        "response": {
            "status_code": status_code,
            "request_id": "req_test",
            "body": {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1711652795,
                "model": "gpt-5.6-luna",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
            },
        },
        "error": None,
    }


class BatchHelpersTests(unittest.TestCase):
    def test_build_chat_body_includes_json_response_format(self):
        body = build_chat_body("gpt-5.6-luna", [{"role": "user", "content": "hi"}])
        self.assertEqual(body["model"], "gpt-5.6-luna")
        self.assertEqual(body["response_format"], {"type": "json_object"})

    def test_build_batch_request_line_schema(self):
        messages = [{"role": "user", "content": "Tweet: hello"}]
        line = build_batch_request_line("tweet-0", "gpt-5.6-luna", messages)
        self.assertEqual(line["custom_id"], "tweet-0")
        self.assertEqual(line["method"], "POST")
        self.assertEqual(line["url"], BATCH_ENDPOINT)
        self.assertEqual(line["body"]["model"], "gpt-5.6-luna")
        self.assertEqual(line["body"]["messages"], messages)
        # single model per file: body model matches request
        self.assertTrue(json.dumps(line))

    def test_write_batch_input_file_and_manifest_roundtrip(self):
        df = pd.DataFrame(
            {
                "id": [11, 22],
                "text": ["I love AI", "I hate bugs"],
                "sentiment": ["positive", "negative"],
                "score_compound": [0.8, -0.6],
            }
        )
        few_shot = [
            {"text": "ex", "sentiment": "neutral", "valence": 0.0},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, "batch_input.jsonl")
            manifest_path = os.path.join(tmpdir, "manifest.json")
            manifest = write_batch_input_file(df, few_shot, "gpt-5.6-luna", in_path)
            save_manifest(manifest, manifest_path)

            self.assertEqual(
                [custom_id_for_row(0), custom_id_for_row(1)], list(manifest.keys())
            )
            self.assertEqual(manifest["tweet-0"]["text"], "I love AI")
            self.assertEqual(manifest["tweet-1"]["true_sentiment"], "negative")

            with open(in_path, encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0]["custom_id"], "tweet-0")
            self.assertEqual(lines[1]["custom_id"], "tweet-1")
            for line in lines:
                self.assertEqual(line["body"]["model"], "gpt-5.6-luna")

            reloaded = load_manifest(manifest_path)
            self.assertEqual(reloaded, manifest)

    def test_extract_content_from_result(self):
        line = _make_chat_result("tweet-0", '{"sentiment": "positive", "valence": 0.8}')
        self.assertIn("positive", extract_content_from_result(line))
        with self.assertRaises(ValueError):
            extract_content_from_result(
                {"custom_id": "tweet-1", "response": None, "error": {"code": "x"}}
            )
        with self.assertRaises(ValueError):
            extract_content_from_result(
                _make_chat_result("tweet-2", "x", status_code=500)
            )

    def test_collect_batch_predictions_with_fallback(self):
        manifest = {
            "tweet-0": {
                "id": 11,
                "text": "I love AI",
                "true_sentiment": "positive",
                "true_score": 0.8,
            },
            "tweet-1": {
                "id": 22,
                "text": "I hate bugs",
                "true_sentiment": "negative",
                "true_score": -0.6,
            },
            "tweet-2": {
                "id": 33,
                "text": "missing",
                "true_sentiment": "neutral",
                "true_score": 0.0,
            },
        }
        lines = [
            _make_chat_result("tweet-0", '{"sentiment": "positive", "valence": 0.9}'),
            {
                "id": "batch_req_1",
                "custom_id": "tweet-1",
                "response": None,
                "error": {"code": "batch_expired", "message": "expired"},
            },
        ]
        (
            y_true_clf,
            y_pred_clf,
            y_true_reg,
            y_pred_reg,
            raw_outputs,
            llm_responses,
            num_errors,
        ) = collect_batch_predictions(manifest, lines)
        # positive=2, negative fallback neutral=1, missing neutral=1
        self.assertEqual(y_true_clf, [2, 0, 1])
        self.assertEqual(y_pred_clf, [2, 1, 1])
        self.assertEqual(y_true_reg, [0.8, -0.6, 0.0])
        self.assertEqual(y_pred_reg[0], 0.9)
        self.assertEqual(len(raw_outputs), 3)
        self.assertEqual(len(llm_responses), 3)
        # one error line + one missing line
        self.assertEqual(num_errors, 2)
        self.assertEqual(raw_outputs[0]["pred_sentiment"], "positive")


if __name__ == "__main__":
    unittest.main()
