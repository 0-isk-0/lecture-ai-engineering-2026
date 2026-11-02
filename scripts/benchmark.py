"""第1回 宿題（発展）: 応答時間とトークン数を、モデルと質問を変えて測る。

使い方:
    uv run python -m scripts.benchmark                       # 既定の3モデル×3質問×3回
    uv run python -m scripts.benchmark --trials 5
    uv run python -m scripts.benchmark --models gemini-3.5-flash-lite,gemini-3.5-flash
    uv run python -m scripts.benchmark --sleep 6             # 無料枠のレート制限がきついとき
    uv run python -m scripts.benchmark --from results/lec01-bench.json   # 測り直さず集計だけ

測るもの:
    - 応答時間（リクエストを投げてから答えが返るまでの実時間）
    - 入力 / 出力 / 思考 / 合計トークン数
    - 出力トークン毎秒（応答時間を出力トークン数で割ったものの逆数）

注意:
    無料枠には1分あたりのリクエスト上限がある。既定では1回ごとに --sleep 秒待ち、
    429（レート制限）が返ったら待ってから1度だけ再試行する。
"""

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from google import genai
from google.genai import errors

from app.ask import build_config, read_usage
from app.config import MODEL, api_error_message, get_api_key

# 既定のモデル。無料枠で使えるものを軽い順に並べている
DEFAULT_MODELS = ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.1-flash-lite"]

# 既定の質問。出力の長さと「モデルが知っているか」が応答時間にどう効くかを見るため、
# 性質の違う3つを使う
DEFAULT_QUESTIONS: dict[str, str] = {
    "short": "LoRAとは何ですか？一文で答えてください。",
    "long": "LoRAの仕組みを、全結合層の重み行列の分解から順に、初学者向けに説明してください。",
    "course": "AIE2026の講座の宿題にある「3レベル」とは何ですか？",
}

# レート制限（429）に当たったときに待つ秒数
RETRY_WAIT_SEC = 20.0


@dataclass
class Record:
    """1回の呼び出しの結果。"""

    model: str
    question_id: str
    trial: int
    ok: bool
    latency_sec: float
    input_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    total_tokens: int = 0
    answer_chars: int = 0
    retried: bool = False
    error: str = ""


@dataclass
class Summary:
    """あるモデル（と質問）の集計結果。"""

    model: str
    question_id: str
    runs: int
    ok_runs: int
    latency_median: float = 0.0
    latency_min: float = 0.0
    latency_max: float = 0.0
    input_tokens_mean: float = 0.0
    output_tokens_mean: float = 0.0
    thought_tokens_mean: float = 0.0
    output_tokens_per_sec: float = 0.0
    errors: list[str] = field(default_factory=list)


def measure(
    client: genai.Client,
    model: str,
    question: str,
    question_id: str,
    trial: int,
    *,
    temperature: float | None,
) -> Record:
    """1回だけ呼び出して、かかった時間とトークン数を記録する。"""
    config = build_config(temperature)
    retried = False

    while True:
        started = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=model, contents=question, config=config
            )
        except errors.APIError as e:
            elapsed = time.perf_counter() - started
            if e.code == 429 and not retried:
                # レート制限は「待てば通る」ので、失敗として残さず1度だけ待ち直す
                retried = True
                time.sleep(RETRY_WAIT_SEC)
                continue
            return Record(
                model=model,
                question_id=question_id,
                trial=trial,
                ok=False,
                latency_sec=round(elapsed, 3),
                retried=retried,
                error=api_error_message(e.code, e.message),
            )

        elapsed = time.perf_counter() - started
        usage = read_usage(response)
        return Record(
            model=model,
            question_id=question_id,
            trial=trial,
            ok=True,
            latency_sec=round(elapsed, 3),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            thought_tokens=usage.thought_tokens,
            total_tokens=usage.total_tokens,
            answer_chars=len(response.text or ""),
            retried=retried,
        )


def run(
    models: list[str],
    questions: dict[str, str],
    trials: int,
    *,
    temperature: float | None,
    sleep_sec: float,
    client: genai.Client | None = None,
) -> list[Record]:
    """モデル×質問×回数ぶん測って、記録を並べて返す。"""
    client = client or genai.Client(api_key=get_api_key())
    records: list[Record] = []
    total = len(models) * len(questions) * trials
    done = 0

    for model in models:
        for question_id, question in questions.items():
            for trial in range(1, trials + 1):
                record = measure(
                    client,
                    model,
                    question,
                    question_id,
                    trial,
                    temperature=temperature,
                )
                records.append(record)
                done += 1
                status = (
                    f"{record.latency_sec:.2f}s / 出力 {record.output_tokens} tok"
                    if record.ok
                    else f"失敗: {record.error}"
                )
                print(f"[{done}/{total}] {model} {question_id} #{trial}  {status}")
                if done < total:
                    time.sleep(sleep_sec)

    return records


def summarize(records: list[Record], *, per_question: bool) -> list[Summary]:
    """記録をモデル単位（または モデル×質問 単位）にまとめる。"""
    groups: dict[tuple[str, str], list[Record]] = {}
    for r in records:
        key = (r.model, r.question_id if per_question else "すべて")
        groups.setdefault(key, []).append(r)

    summaries: list[Summary] = []
    for (model, question_id), rows in groups.items():
        ok_rows = [r for r in rows if r.ok]
        summary = Summary(
            model=model,
            question_id=question_id,
            runs=len(rows),
            ok_runs=len(ok_rows),
            errors=sorted({r.error for r in rows if not r.ok}),
        )
        if ok_rows:
            latencies = [r.latency_sec for r in ok_rows]
            summary.latency_median = round(statistics.median(latencies), 3)
            summary.latency_min = round(min(latencies), 3)
            summary.latency_max = round(max(latencies), 3)
            summary.input_tokens_mean = round(
                statistics.mean(r.input_tokens for r in ok_rows), 1
            )
            summary.output_tokens_mean = round(
                statistics.mean(r.output_tokens for r in ok_rows), 1
            )
            summary.thought_tokens_mean = round(
                statistics.mean(r.thought_tokens for r in ok_rows), 1
            )
            # 「速さ」は応答時間だけだと出力が長い質問で不利になるので、
            # 出力トークン数で割った値も見る
            seconds = sum(latencies)
            produced = sum(r.output_tokens + r.thought_tokens for r in ok_rows)
            if seconds > 0:
                summary.output_tokens_per_sec = round(produced / seconds, 1)
        summaries.append(summary)

    summaries.sort(key=lambda s: (s.model, s.question_id))
    return summaries


def render_markdown(summaries: list[Summary], *, per_question: bool) -> str:
    """集計結果を answer.md に貼れる表にする。"""
    header = ["モデル"]
    if per_question:
        header.append("質問")
    header += [
        "成功",
        "応答時間 中央値(s)",
        "最小〜最大(s)",
        "入力tok",
        "出力tok",
        "思考tok",
        "出力tok/s",
    ]

    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for s in summaries:
        cells = [s.model]
        if per_question:
            cells.append(s.question_id)
        cells += [
            f"{s.ok_runs}/{s.runs}",
            f"{s.latency_median:.2f}",
            f"{s.latency_min:.2f}〜{s.latency_max:.2f}",
            f"{s.input_tokens_mean:.0f}",
            f"{s.output_tokens_mean:.0f}",
            f"{s.thought_tokens_mean:.0f}",
            f"{s.output_tokens_per_sec:.1f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")

    failed = [s for s in summaries if s.errors]
    if failed:
        lines.append("")
        lines.append("失敗した呼び出し:")
        for s in failed:
            for e in s.errors:
                label = f"{s.model} / {s.question_id}" if per_question else s.model
                lines.append(f"- {label}: {e}")

    return "\n".join(lines)


def load_records(path: Path) -> list[Record]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Record(**row) for row in data["records"]]


def save_records(path: Path, records: list[Record], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "records": [asdict(r) for r in records]}
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="応答時間とトークン数を比べる")
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help=f"カンマ区切り（既定: {','.join(DEFAULT_MODELS)}）",
    )
    parser.add_argument("--trials", type=int, default=3, help="同じ条件で何回ずつ測るか")
    parser.add_argument("-t", "--temperature", type=float, default=None)
    parser.add_argument(
        "--sleep",
        type=float,
        default=4.0,
        help="1回ごとに待つ秒数。無料枠の1分あたりの上限に合わせる（既定: 4.0）",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/lec01-bench.json"),
        help="測った生データの保存先",
    )
    parser.add_argument(
        "--from",
        dest="from_path",
        type=Path,
        default=None,
        help="測り直さず、保存済みの JSON から集計し直す",
    )
    parser.add_argument(
        "--per-question", action="store_true", help="質問ごとに分けて集計する"
    )
    args = parser.parse_args()

    if args.from_path:
        records = load_records(args.from_path)
        print(f"[読み込み] {args.from_path}（{len(records)} 件）\n")
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        print(f"[既定モデル] {MODEL}")
        print(f"[測るモデル] {', '.join(models)}")
        print(f"[質問] {', '.join(DEFAULT_QUESTIONS)}")
        print(f"[回数] 各 {args.trials} 回 / 1回ごとに {args.sleep} 秒待つ\n")
        records = run(
            models,
            DEFAULT_QUESTIONS,
            args.trials,
            temperature=args.temperature,
            sleep_sec=args.sleep,
            client=None,
        )
        save_records(
            args.out,
            records,
            {
                "models": models,
                "questions": DEFAULT_QUESTIONS,
                "trials": args.trials,
                "temperature": args.temperature,
                "sleep_sec": args.sleep,
            },
        )
        print(f"\n[保存] {args.out}")

    print("\n## モデルごと\n")
    print(render_markdown(summarize(records, per_question=False), per_question=False))

    if args.per_question or args.from_path:
        print("\n## モデル×質問ごと\n")
        print(render_markdown(summarize(records, per_question=True), per_question=True))


if __name__ == "__main__":
    main()
