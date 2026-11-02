"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask                         # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask -t 1.5 "LoRAとは？"      # temperature を変えて答えのばらつきを見る
    uv run python -m app.ask --chat                  # 会話履歴を保ったまま続けて質問する

ツールを持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
from dataclasses import dataclass

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"

# --chat のときに使えるコマンド
EXIT_COMMANDS = ("exit", "quit", ":q")
RESET_COMMAND = ":reset"


@dataclass(frozen=True)
class Usage:
    """1回の呼び出しで使ったトークン数。

    thought は「思考モデル」が内部の推論に使った分で、出力（candidates）とは別に数えられる。
    無いモデルでは 0 になる。
    """

    input_tokens: int
    output_tokens: int
    thought_tokens: int
    total_tokens: int

    def line(self) -> str:
        parts = [f"入力 {self.input_tokens}", f"出力 {self.output_tokens}"]
        if self.thought_tokens:
            parts.append(f"思考 {self.thought_tokens}")
        parts.append(f"合計 {self.total_tokens}")
        return "[tokens] " + " / ".join(parts)


def read_usage(response: types.GenerateContentResponse) -> Usage:
    """response.usage_metadata を Usage にする。項目が返らないときは 0 として扱う。"""
    meta = response.usage_metadata

    def count(name: str) -> int:
        if meta is None:
            return 0
        return getattr(meta, name, None) or 0

    return Usage(
        input_tokens=count("prompt_token_count"),
        output_tokens=count("candidates_token_count"),
        thought_tokens=count("thoughts_token_count"),
        total_tokens=count("total_token_count"),
    )


def build_config(temperature: float | None) -> types.GenerateContentConfig | None:
    """temperature を指定したときだけ config を作る（未指定ならモデルの既定値に任せる）。"""
    if temperature is None:
        return None
    return types.GenerateContentConfig(temperature=temperature)


def user_turn(text: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part(text=text)])


def model_turn(text: str) -> types.Content:
    return types.Content(role="model", parts=[types.Part(text=text)])


def ask(
    question: str,
    *,
    client: genai.Client | None = None,
    model: str = MODEL,
    temperature: float | None = None,
) -> tuple[str, Usage]:
    """1回だけ質問して、答えと使ったトークン数を返す。"""
    client = client or genai.Client(api_key=get_api_key())
    response = client.models.generate_content(
        model=model,
        contents=question,
        config=build_config(temperature),
    )
    return response.text or "", read_usage(response)


def chat(
    client: genai.Client,
    history: list[types.Content],
    question: str,
    *,
    model: str = MODEL,
    temperature: float | None = None,
) -> tuple[str, Usage]:
    """これまでの会話に質問を足して送り、答えを history に追記する。

    LLM 自体は前回のやり取りを覚えていない。毎回 history ごと送り直すことで
    「続けて質問できる」ように見せている。だから往復を重ねるほど入力トークンが増える。
    """
    history.append(user_turn(question))
    response = client.models.generate_content(
        model=model,
        contents=list(history),  # このあと答えを足すので、送る分はコピーを渡す
        config=build_config(temperature),
    )
    answer = response.text or ""
    history.append(model_turn(answer))
    return answer, read_usage(response)


def run_once(args: argparse.Namespace) -> None:
    print(f"[model] {args.model}")
    print(f"[質問] {args.question}\n")
    answer, usage = ask(
        args.question, model=args.model, temperature=args.temperature
    )
    print(answer)
    print(f"\n{usage.line()}")


def run_chat(args: argparse.Namespace) -> None:
    """会話履歴を保ったまま続けて質問する（第1回 宿題・実装）。"""
    client = genai.Client(api_key=get_api_key())
    history: list[types.Content] = []

    print(f"[model] {args.model}")
    print(f"続けて質問できます。終了は {' / '.join(EXIT_COMMANDS)}、履歴を消すのは {RESET_COMMAND}\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            continue
        if question in EXIT_COMMANDS:
            return
        if question == RESET_COMMAND:
            history.clear()
            print("履歴を消しました。\n")
            continue

        try:
            answer, usage = chat(
                client,
                history,
                question,
                model=args.model,
                temperature=args.temperature,
            )
        except errors.APIError as e:
            # 履歴に質問だけが残ると、次の送信で user が連続してしまうので戻す
            history.pop()
            print(api_error_message(e.code, e.message) + "\n")
            continue

        print(f"\n{answer}")
        print(f"{usage.line()} / 往復 {len(history) // 2}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--model", default=MODEL, help=f"使うモデル（既定: {MODEL}、環境変数 AIE_MODEL）"
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=None,
        help="0 に近いほど答えが安定し、大きいほどばらつく（未指定ならモデルの既定値）",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="会話履歴を保ったまま続けて質問する",
    )
    args = parser.parse_args()

    try:
        if args.chat:
            run_chat(args)
        else:
            run_once(args)
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
