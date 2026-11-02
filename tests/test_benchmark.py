"""第1回の benchmark のテスト（LLMもAPIキーも使わない）。

集計と表の組み立ては、測定と切り離して検証できるようにしてある。
"""

import json

from scripts import benchmark


def record(model: str, question_id: str, trial: int, latency: float, **kwargs):
    base = dict(
        ok=True, input_tokens=10, output_tokens=100, thought_tokens=0, total_tokens=110
    )
    base.update(kwargs)
    return benchmark.Record(
        model=model,
        question_id=question_id,
        trial=trial,
        latency_sec=latency,
        **base,
    )


def test_summarize_groups_by_model():
    records = [
        record("a", "short", 1, 1.0),
        record("a", "long", 2, 3.0),
        record("b", "short", 1, 2.0),
    ]
    summaries = benchmark.summarize(records, per_question=False)
    assert [s.model for s in summaries] == ["a", "b"]
    assert summaries[0].runs == 2
    assert summaries[0].latency_median == 2.0
    assert summaries[0].latency_min == 1.0
    assert summaries[0].latency_max == 3.0


def test_summarize_can_split_by_question():
    records = [record("a", "short", 1, 1.0), record("a", "long", 1, 5.0)]
    summaries = benchmark.summarize(records, per_question=True)
    assert [(s.model, s.question_id) for s in summaries] == [("a", "long"), ("a", "short")]


def test_summarize_ignores_failed_runs_but_counts_them():
    records = [
        record("a", "short", 1, 1.0),
        record("a", "short", 2, 0.5, ok=False, error="レート制限に達しました。"),
    ]
    (summary,) = benchmark.summarize(records, per_question=False)
    assert (summary.runs, summary.ok_runs) == (2, 1)
    assert summary.latency_median == 1.0  # 失敗した 0.5 秒は混ぜない
    assert summary.errors == ["レート制限に達しました。"]


def test_summarize_handles_all_runs_failing():
    records = [record("a", "short", 1, 0.2, ok=False, error="APIエラー（500）")]
    (summary,) = benchmark.summarize(records, per_question=False)
    assert summary.ok_runs == 0
    assert summary.latency_median == 0.0
    assert summary.output_tokens_per_sec == 0.0


def test_output_tokens_per_sec_counts_thought_tokens():
    records = [record("a", "short", 1, 2.0, output_tokens=100, thought_tokens=100)]
    (summary,) = benchmark.summarize(records, per_question=False)
    assert summary.output_tokens_per_sec == 100.0


def test_render_markdown_has_one_row_per_summary():
    records = [record("a", "short", 1, 1.0), record("b", "short", 1, 2.0)]
    table = benchmark.render_markdown(
        benchmark.summarize(records, per_question=False), per_question=False
    )
    lines = table.splitlines()
    assert lines[0].startswith("| モデル |")
    assert len(lines) == 4  # 見出し + 区切り + 2行
    assert "質問" not in lines[0]


def test_render_markdown_lists_errors():
    records = [record("a", "short", 1, 0.1, ok=False, error="APIキーかモデル名に問題があります（404）")]
    table = benchmark.render_markdown(
        benchmark.summarize(records, per_question=False), per_question=False
    )
    assert "失敗した呼び出し" in table
    assert "404" in table


def test_records_survive_a_save_and_load_round_trip(tmp_path):
    records = [record("a", "short", 1, 1.25)]
    path = tmp_path / "out" / "bench.json"
    benchmark.save_records(path, records, {"trials": 1})

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["meta"]["trials"] == 1

    assert benchmark.load_records(path) == records


def test_default_questions_cover_short_long_and_course_specific():
    assert set(benchmark.DEFAULT_QUESTIONS) == {"short", "long", "course"}
    assert "AIE2026" in benchmark.DEFAULT_QUESTIONS["course"]
