"""Run the assignment's six RAG questions against a real Ollama endpoint."""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag import UNKNOWN, RagError, build_index, generate_answer, retrieve


CASES = [
    ('Answerable 1', 'Where can I buy souvenirs in the city?', 'G01'),
    ('Answerable 2', 'Which walking street links City Hall and Bach Dang Wharf?', 'G02'),
    ('Answerable 3', 'What are the Cu Chi Tunnels?', 'G10'),
    ('Paraphrase', 'Where can I explore the Chinese quarter?', 'G06'),
    ('Outside guide 1', 'How much is the ticket for Ben Thanh Market tomorrow?', None),
    ('Outside guide 2', 'At what time does Saigon Opera House open tomorrow?', None),
]


def automated_result(expected_id: str | None, retrieved_ids: list[str], answer: str) -> tuple[str, str]:
    """Return a conservative automated check; final semantic review is manual."""
    if expected_id is None:
        return ('Yes', '—') if answer == UNKNOWN else ('No', 'generation')
    if expected_id not in retrieved_ids:
        return 'No', 'retrieval'
    if answer == UNKNOWN or f'[{expected_id}]' not in answer:
        return 'No', 'generation'
    return 'Review', '—'


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--ollama-url', required=True, help='Ollama base URL, without /api')
    parser.add_argument('--top-k', type=int, default=3)
    args = parser.parse_args()

    try:
        entries, vectors = build_index(args.ollama_url, str(PROJECT_ROOT / 'knowledge_base.json'))
        print('| Type | Question | Expected | Retrieved IDs (scores) | Automated check | Failure step |')
        print('|---|---|---|---|---|---|')
        for case_type, question, expected_id in CASES:
            sources = retrieve(args.ollama_url, question, entries, vectors, k=args.top_k)
            answer = generate_answer(args.ollama_url, question, sources)
            retrieved_ids = [source['id'] for source in sources]
            retrieved = ', '.join(f"{source['id']} ({source['score']:.3f})" for source in sources)
            check, failure = automated_result(expected_id, retrieved_ids, answer)
            safe_answer = answer.replace('|', '\\|').replace('\n', ' ')
            print(f'| {case_type} | {question} | {expected_id or "None"} | {retrieved} | {check} | {failure} |')
            print(f'  Answer: {safe_answer}')
    except (RagError, OSError, ValueError) as exc:
        print(f'Evaluation failed: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
