"""第2回 宿題（発展）: 用語集から SFT 用の学習データと評価データを作る。

教材の scripts/make_sft_dataset.py は学習データ（messages 形式の JSONL）だけを出力する。
発展レベルでは「評価用の質問と正解を自分で用意する」ことが要求されるので、
ここでは学習データに加えて、評価用の test と gold も同時に作る。

出力（既定では --out-dir に 3 ファイル）:
    glossary_sft.jsonl    学習データ。各行 {"messages": [...]}。ノートブックの to_chat と同じ形
    glossary_test.jsonl   評価データ。各行 {"question", "target", "needs_review", "bucket"}
    glossary_gold.json    採点の正解。{質問: {"needs_review": bool, "must_include": [...]}}

使い方:
    uv run python scripts/make_glossary_sft.py --glossary data/glossary.json --out-dir .

設計:
- 学習の入力は質問だけで、用語集の本文はプロンプトに入れない。
  用語集を渡してしまうと「文脈から写す」課題になり、
  重みに知識が入ったかどうかを測れなくなる（検索との比較ができなくなる）。
- 出力形式は演習と同じ {"answer": 文字列, "needs_review": 真偽値} の JSON 1 行に揃える。
- 評価は 3 種類の質問に分ける。
    known      用語集にある語を、学習に使っていない言い回しで聞く
    paraphrase 用語集にある語を、別名（aliases）で聞く
    unknown    用語集に無い語を聞く。needs_review=true が正解
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

# 学習に使う質問の言い回し。test ではこれらを使わない（学習した文がそのまま評価に出ないようにする）
TRAIN_TEMPLATES = [
    "「{term}」とは何ですか？",
    "{term}について簡単に説明してください。",
]
# 評価にだけ使う言い回し
TEST_TEMPLATE = "{term}とはどういうものか説明してください。"
PARAPHRASE_TEMPLATE = "{term}って何のことですか？"

UNKNOWN_ANSWER = "この用語集には記載がありません。確認が必要です。"
UNKNOWN_MUST_INCLUDE = ["確認"]

# 用語集（58 語）に無い AI / 開発まわりの語。実在する語をあえて選び、
# 「もっともらしいので断定しやすい」状況を作る。needs_review=true が正解。
UNKNOWN_TERMS = [
    "RAG（検索拡張生成）",
    "Transformer",
    "知識蒸留",
    "ベクトルデータベース",
    "埋め込みベクトル",
    "強化学習",
    "RLHF",
    "ビームサーチ",
    "混合エキスパート",
    "プロンプトインジェクション",
    "アテンション機構",
    "トークナイザ",
    "グラディエントチェックポインティング",
    "early stopping",
    "データ拡張",
    "交差検証",
]

SYSTEM = (
    "あなたはAIエンジニアリング実践講座の用語解説係です。"
    "質問された用語を知っている場合だけ、その定義を answer に書き、needs_review を false にします。"
    "知らない用語は answer に『この用語集には記載がありません。確認が必要です。』と書き、"
    "needs_review を true にします。"
    '必ず JSON 1 行で答えてください。形式: {"answer": 文字列, "needs_review": 真偽値}'
)

# must_include を取り出すときに拾ってしまいやすい、どの定義にも出る一般語
STOPWORDS = {
    "こと", "もの", "ため", "とき", "場合", "ここ", "それ", "これ", "ある", "する",
    "本講座", "講座", "モデル", "使う", "多い", "必要", "できる", "など",
}
# 3 文字以上の漢字・カタカナ・英数の連なり
CHUNK = re.compile(r"[一-龥]{3,}|[ァ-ヴー]{3,}|[A-Za-z][A-Za-z0-9.\-]{2,}")


def first_sentence(text: str) -> str:
    """定義の第1文だけを学習の答えにする。

    定義は 100〜200 字あり、生成の上限（max_new_tokens）に収まらないことがある。
    第1文に切ると、形式と内容の採点が「長さの都合で落ちる」ことを避けられる。
    """
    head = text.split("。", 1)[0]
    return head + "。"


def must_include(term: str, answer: str, k: int = 2) -> list[str]:
    """答えに含まれるべき語を、定義から機械的に選ぶ。

    用語そのものは除く（質問文の写しで通ってしまうため）。
    長い語ほど固有性が高いので、長さ順に k 個取る。
    """
    seen: dict[str, None] = {}
    for chunk in CHUNK.findall(answer):
        if chunk in STOPWORDS or chunk in term or term in chunk:
            continue
        seen.setdefault(chunk, None)
    picked = sorted(seen, key=len, reverse=True)[:k]
    if picked:
        return picked
    # 漢字・カタカナの塊が無い短い定義（「作る→測る→直す…」など）は、先頭 8 文字を必須にする。
    # 空のままにすると needs_review の一致だけで内容が ○ になり、採点が緩くなる。
    return [answer[:8]]


def chat_record(question: str, answer: str, needs_review: bool) -> dict:
    target = json.dumps({"answer": answer, "needs_review": needs_review}, ensure_ascii=False)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
            {"role": "assistant", "content": target},
        ]
    }


def build(glossary_path: Path, seed: int, per_bucket: int) -> tuple[list[dict], list[dict], dict]:
    terms = json.loads(glossary_path.read_text(encoding="utf-8"))["terms"]
    rng = random.Random(seed)

    train: list[dict] = []
    # 評価は種類ごとに集めてから同数ずつ抜き出す。
    # 全部（用語 58 語ぶん）を評価すると、学習前後の 2 回で T4 の生成時間が十数分かかる。
    pools: dict[str, list[dict]] = {"known": [], "paraphrase": [], "unknown": []}
    gold_all: dict[str, dict] = {}

    # 「知らない」と答える振る舞いも学習させる。これを入れないと、
    # 学習後は全部 needs_review=false になり、未知語をすべて断定するようになる。
    unknown_for_train = UNKNOWN_TERMS[: len(UNKNOWN_TERMS) // 2]
    unknown_for_test = UNKNOWN_TERMS[len(UNKNOWN_TERMS) // 2 :]

    for entry in terms:
        term = entry["term"]
        answer = first_sentence(entry["definition"])
        for template in TRAIN_TEMPLATES:
            train.append(chat_record(template.format(term=term), answer, False))

        # known: 学習に使っていない言い回しで、同じ語を聞く
        q = TEST_TEMPLATE.format(term=term)
        pools["known"].append({"question": q, "target": answer, "needs_review": False, "bucket": "known"})
        gold_all[q] = {"needs_review": False, "must_include": must_include(term, answer)}

        # paraphrase: 別名で聞く。学習では正式名称しか見ていない語を、別の呼び方で引けるか
        for alias in entry["aliases"]:
            if alias == term:
                continue
            qa = PARAPHRASE_TEMPLATE.format(term=alias)
            if qa in gold_all:
                continue
            pools["paraphrase"].append({"question": qa, "target": answer, "needs_review": False, "bucket": "paraphrase"})
            gold_all[qa] = {"needs_review": False, "must_include": must_include(alias, answer)}

    for term in unknown_for_train:
        for template in TRAIN_TEMPLATES:
            train.append(chat_record(template.format(term=term), UNKNOWN_ANSWER, True))

    for term in unknown_for_test:
        q = TEST_TEMPLATE.format(term=term)
        pools["unknown"].append({"question": q, "target": UNKNOWN_ANSWER, "needs_review": True, "bucket": "unknown"})
        gold_all[q] = {"needs_review": True, "must_include": UNKNOWN_MUST_INCLUDE}

    test: list[dict] = []
    for bucket, pool in pools.items():
        picked = sorted(pool, key=lambda r: r["question"])
        rng.shuffle(picked)
        test.extend(picked[:per_bucket])
    test.sort(key=lambda r: (r["bucket"], r["question"]))
    gold = {r["question"]: gold_all[r["question"]] for r in test}

    rng.shuffle(train)

    train_questions = {r["messages"][1]["content"] for r in train}
    overlap = train_questions & {r["question"] for r in test}
    if overlap:
        raise SystemExit(f"学習データと評価データで質問が重複しています: {sorted(overlap)[:3]}")

    return train, test, gold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--glossary", type=Path, default=Path("data/glossary.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--per-bucket", type=int, default=14,
        help="評価データを種類ごとに何問にするか（known / paraphrase / unknown）。既定 14 で合計 42 問",
    )
    args = parser.parse_args()

    if not args.glossary.exists():
        raise SystemExit(f"用語集が見つかりません: {args.glossary}")

    train, test, gold = build(args.glossary, args.seed, args.per_bucket)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    (args.out_dir / "glossary_sft.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n", encoding="utf-8"
    )
    (args.out_dir / "glossary_test.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in test) + "\n", encoding="utf-8"
    )
    (args.out_dir / "glossary_gold.json").write_text(
        json.dumps(gold, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    buckets: dict[str, int] = {}
    for r in test:
        buckets[r["bucket"]] = buckets.get(r["bucket"], 0) + 1
    print(f"train={len(train)} test={len(test)} gold={len(gold)}", file=sys.stderr)
    print(f"test の内訳: {buckets}", file=sys.stderr)
    print(f"出力先: {args.out_dir.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
