"""第1回の ask のテスト（LLMもAPIキーも使わない）。

google-genai のレスポンスの形だけを真似た偽クライアントを使う。
"""

from dataclasses import dataclass

import pytest

from app import ask


@dataclass
class FakeUsage:
    prompt_token_count: int = 0
    candidates_token_count: int = 0
    thoughts_token_count: int = 0
    total_token_count: int = 0


@dataclass
class FakeResponse:
    text: str | None = "答え"
    usage_metadata: FakeUsage | None = None


class FakeModels:
    """generate_content に渡された contents を記録するだけのモデル。"""

    def __init__(self, responses: list[FakeResponse]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def generate_content(self, *, model, contents, config=None):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, responses: list[FakeResponse]):
        self.models = FakeModels(responses)


def test_read_usage_picks_up_each_token_count():
    response = FakeResponse(
        usage_metadata=FakeUsage(
            prompt_token_count=11,
            candidates_token_count=22,
            thoughts_token_count=33,
            total_token_count=66,
        )
    )
    usage = ask.read_usage(response)
    assert (usage.input_tokens, usage.output_tokens) == (11, 22)
    assert (usage.thought_tokens, usage.total_tokens) == (33, 66)


def test_read_usage_treats_missing_metadata_as_zero():
    usage = ask.read_usage(FakeResponse(usage_metadata=None))
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (0, 0, 0)


def test_usage_line_hides_thought_tokens_when_zero():
    line = ask.Usage(input_tokens=10, output_tokens=5, thought_tokens=0, total_tokens=15).line()
    assert "思考" not in line
    assert "入力 10" in line and "出力 5" in line

    with_thought = ask.Usage(
        input_tokens=10, output_tokens=5, thought_tokens=7, total_tokens=22
    ).line()
    assert "思考 7" in with_thought


def test_build_config_is_none_without_temperature():
    assert ask.build_config(None) is None
    assert ask.build_config(0.2).temperature == pytest.approx(0.2)


def test_ask_returns_answer_and_usage():
    client = FakeClient([FakeResponse(text="LoRAは…", usage_metadata=FakeUsage(total_token_count=40))])
    answer, usage = ask.ask("LoRAとは？", client=client, model="fake-model")
    assert answer == "LoRAは…"
    assert usage.total_tokens == 40
    assert client.models.calls[0]["model"] == "fake-model"


def test_ask_turns_missing_text_into_empty_string():
    client = FakeClient([FakeResponse(text=None)])
    answer, _ = ask.ask("質問", client=client)
    assert answer == ""


def test_chat_keeps_history_and_resends_it():
    client = FakeClient([FakeResponse(text="1回目"), FakeResponse(text="2回目")])
    history: list = []

    ask.chat(client, history, "最初の質問")
    assert [c.role for c in history] == ["user", "model"]

    ask.chat(client, history, "続きの質問")
    assert [c.role for c in history] == ["user", "model", "user", "model"]

    # 2回目の呼び出しでは、1回目のやり取りを含めた履歴ごと送っている
    sent = client.models.calls[1]["contents"]
    assert len(sent) == 3
    assert sent[0].parts[0].text == "最初の質問"
    assert sent[1].parts[0].text == "1回目"
    assert sent[2].parts[0].text == "続きの質問"


def test_default_question_is_course_specific():
    assert "3レベル" in ask.DEFAULT_QUESTION
