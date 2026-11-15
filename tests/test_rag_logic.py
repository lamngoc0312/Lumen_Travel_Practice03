"""Offline unit checks. Not an end-to-end Ollama/Firebase test."""
import json
import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rag


class TestRAGLogic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guide_path = Path(__file__).resolve().parents[1] / 'knowledge_base.json'

    def test_guide_size_and_unique_ids(self):
        guide = rag.load_guide(str(self.guide_path))
        self.assertTrue(12 <= len(guide) <= 20)
        self.assertEqual(len({e['id'] for e in guide}), len(guide))

    def test_guide_schema_required_for_every_entry(self):
        guide = rag.load_guide(str(self.guide_path))
        for entry in guide:
            self.assertRegex(entry['id'], r'^G\d{2}$')
            self.assertTrue(entry['title'].strip())
            self.assertTrue(entry['text'].strip())
            self.assertTrue(entry['source'].startswith(('http://', 'https://')))
            self.assertTrue(entry['category'].strip())
            self.assertTrue(entry['tags'])

    def test_invalid_guide_reports_entry_and_field(self):
        guide = json.loads(self.guide_path.read_text(encoding='utf-8'))
        guide[0]['tags'] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_path = Path(temp_dir) / 'knowledge_base.json'
            invalid_path.write_text(json.dumps(guide), encoding='utf-8')
            with self.assertRaisesRegex(rag.RagError, r'G01.*tags'):
                rag.load_guide(str(invalid_path))

    def test_cosine(self):
        self.assertAlmostEqual(rag.cosine_similarity([1, 0], [1, 0]), 1)
        self.assertAlmostEqual(rag.cosine_similarity([1, 0], [0, 1]), 0)

    def test_retrieval_order(self):
        entries = [{'id': 'G01'}, {'id': 'G02'}, {'id': 'G03'}]
        vectors = [[1, 0], [0, 1], [-1, 0]]
        with patch.object(rag, 'embed_texts', return_value=[[1, 0]]):
            hits = rag.retrieve('https://example.invalid', 'question', entries, vectors, k=2)
        self.assertEqual([x['id'] for x in hits], ['G01', 'G02'])

    def test_unknown_when_no_citation(self):
        with patch.object(rag, '_post', return_value={'message': {'content': 'The answer is 4 PM.'}}):
            answer = rag.generate_answer('https://example.invalid', 'Question', [{'id': 'G01', 'title':'T', 'text':'No time listed', 'score':0.9}])
        self.assertEqual(answer, rag.UNKNOWN)

    def test_unknown_when_similarity_is_below_threshold(self):
        with patch.object(rag, '_post') as post:
            answer = rag.generate_answer(
                'https://example.invalid', 'Question',
                [{'id': 'G01', 'title': 'T', 'text': 'Context', 'score': 0.3}],
                min_similarity=0.5,
            )
        self.assertEqual(answer, rag.UNKNOWN)
        post.assert_not_called()

    def test_answer_with_retrieved_citation_is_kept(self):
        expected = 'Ben Thanh Market has souvenir stalls [G01].'
        with patch.object(rag, '_post', return_value={'message': {'content': expected}}):
            answer = rag.generate_answer(
                'https://example.invalid', 'Question',
                [{'id': 'G01', 'title': 'Market', 'text': 'Souvenir stalls', 'score': 0.9}],
            )
        self.assertEqual(answer, expected)

    def test_qwen_thinking_is_removed_from_visible_answer(self):
        raw = 'Internal reasoning that must not be shown.</think>\nBen Thanh Market [G01].'
        with patch.object(rag, '_post', return_value={'message': {'content': raw}}):
            answer = rag.generate_answer(
                'https://example.invalid', 'Question',
                [{'id': 'G01', 'title': 'Market', 'text': 'Souvenir stalls', 'score': 0.9}],
            )
        self.assertEqual(answer, 'Ben Thanh Market [G01].')

    def test_answer_with_unretrieved_citation_is_rejected(self):
        with patch.object(rag, '_post', return_value={'message': {'content': 'Answer [G99].'}}):
            answer = rag.generate_answer(
                'https://example.invalid', 'Question',
                [{'id': 'G01', 'title': 'Market', 'text': 'Context', 'score': 0.9}],
            )
        self.assertEqual(answer, rag.UNKNOWN)

    def test_invalid_embedding_vectors_are_rejected(self):
        with patch.object(rag, '_post', return_value={'embeddings': [[1.0, 2.0], [1.0]]}):
            with self.assertRaisesRegex(rag.RagError, 'invalid vectors'):
                rag.embed_texts('https://example.invalid', ['one', 'two'])

    def test_ollama_connection_error_is_readable(self):
        with patch.object(rag.requests, 'post', side_effect=rag.requests.ConnectionError):
            with self.assertRaisesRegex(rag.RagError, 'Cannot reach Ollama'):
                rag._post('https://example.invalid', '/api/embed', {})


if __name__ == '__main__':
    unittest.main()
