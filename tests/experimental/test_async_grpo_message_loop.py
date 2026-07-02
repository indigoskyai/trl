# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio

import trl.experimental.async_grpo.async_rollout_worker as worker
from trl.experimental.async_grpo.async_rollout_worker import MessageRolloutLoop

from ..testing_utils import TrlTestCase


# A tool-calling assistant turn keeps the loop going; a plain turn ends it.
_TOOL_CALL = {
    "role": "assistant",
    "content": "",
    "tool_calls": [{"type": "function", "function": {"name": "t", "arguments": {}}}],
}
_FINAL = {"role": "assistant", "content": "done"}


def _run(monkeypatch, *, prompt_ids, turns, assistants, fork_threshold=1024, max_iters=None):
    """Drive MessageRolloutLoop._generate_one on scripted per-turn fixtures.

    prompt_ids: list of the token list `apply_chat_template` returns each turn. turns: list of (turn_ids, logprobs)
    `_generate_one_turn` returns each turn. assistants: list of the message `parse_response` returns each turn.
    """
    pq, tq, aq = list(prompt_ids), list(turns), list(assistants)
    monkeypatch.setattr(worker, "parse_response", lambda tokenizer, ids, prefix=None: aq.pop(0))

    class _StubTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return pq.pop(0)

    loop = object.__new__(MessageRolloutLoop)  # skip the heavy __init__; set only what _generate_one reads
    loop.tokenizer = _StubTokenizer()
    loop.tools = []
    loop.chat_template = None
    loop.chat_template_kwargs = {}
    loop.max_tool_calling_iterations = max_iters
    loop._fork_threshold_tokens = fork_threshold

    async def _generate_one_turn(prompt_ids):
        return tq.pop(0)

    loop._generate_one_turn = _generate_one_turn
    loop._execute_tool_calls = lambda tool_calls, tool_dict: ([{"role": "tool", "name": "t", "content": "ok"}], 1, 0)

    # _generate_one returns (prompt_ids, completion, completion_ids, sequences, n_calls, n_failures);
    # drop the leading prompt_ids, which the tests below don't assert on.
    _prompt_ids, *rest = asyncio.run(loop._generate_one([{"role": "user", "content": "hi"}], {}))
    return tuple(rest)


class TestMessageRolloutLoop(TrlTestCase):
    def test_single_turn_no_tool_call(self, monkeypatch):
        completion, completion_ids, sequences, n_calls, n_failures = _run(
            monkeypatch,
            prompt_ids=[[1, 2, 3]],
            turns=[([10, 11], [-0.1, -0.2])],
            assistants=[_FINAL],
        )
        assert len(sequences) == 1
        assert sequences[0].input_ids == [1, 2, 3, 10, 11]
        assert sequences[0].completion_mask == [0, 0, 0, 1, 1]
        assert sequences[0].old_log_probs == [0.0, 0.0, 0.0, -0.1, -0.2]
        assert completion_ids == [10, 11]
        assert [m["role"] for m in completion] == ["assistant"]
        assert n_calls == 0 and n_failures == 0

    def test_clean_two_turns_stay_one_row(self, monkeypatch):
        # Turn 2's re-tokenized prompt starts with what we held (gen tokens + tool tokens) -> CLEAN.
        completion, completion_ids, sequences, n_calls, n_failures = _run(
            monkeypatch,
            prompt_ids=[[1, 2, 3], [1, 2, 3, 10, 11, 20, 21]],
            turns=[([10, 11], [-0.1, -0.2]), ([30, 31], [-0.3, -0.4])],
            assistants=[_TOOL_CALL, _FINAL],
        )
        assert len(sequences) == 1
        assert sequences[0].input_ids == [1, 2, 3, 10, 11, 20, 21, 30, 31]
        assert sequences[0].completion_mask == [0, 0, 0, 1, 1, 0, 0, 1, 1]  # prompt=0, gen=1, tool=0, gen=1
        assert sequences[0].old_log_probs == [0.0, 0.0, 0.0, -0.1, -0.2, 0.0, 0.0, -0.3, -0.4]
        assert completion_ids == [10, 11, 30, 31]  # generated tokens only, both turns
        assert [m["role"] for m in completion] == ["assistant", "tool", "assistant"]
        assert n_calls == 1 and n_failures == 0

    def test_history_rewrite_forks_into_two_rows(self, monkeypatch):
        # Turn 2's prompt diverges inside turn 1's answer and the new turn is >= fork_threshold -> FORK.
        _, _, sequences, _, _ = _run(
            monkeypatch,
            prompt_ids=[[1, 2, 3], [1, 2, 3, 99, 88, 77]],
            turns=[([10, 11, 12, 13], [-0.1] * 4), ([30, 31, 32], [-0.2] * 3)],
            assistants=[_TOOL_CALL, _FINAL],
            fork_threshold=2,
        )
        assert len(sequences) == 2
        assert sequences[0].input_ids == [1, 2, 3, 10, 11, 12, 13]
        assert sequences[0].completion_mask == [0, 0, 0, 1, 1, 1, 1]
        assert sequences[1].input_ids == [1, 2, 3, 99, 88, 77, 30, 31, 32]
        assert sequences[1].completion_mask == [0, 0, 0, 0, 0, 0, 1, 1, 1]  # 6 context (rewritten history) + 3 gen
        # Every generated token is trained exactly once across the rows.
        assert sum(sum(s.completion_mask) for s in sequences) == 4 + 3

    def test_max_tool_calling_iterations_caps_turns(self, monkeypatch):
        # max_iters=0: even though turn 1 is a tool call, the loop breaks before executing it.
        completion, _, sequences, n_calls, _ = _run(
            monkeypatch,
            prompt_ids=[[1, 2, 3]],
            turns=[([10, 11], [-0.1, -0.2])],
            assistants=[_TOOL_CALL],
            max_iters=0,
        )
        assert len(sequences) == 1
        assert [m["role"] for m in completion] == ["assistant"]  # no tool message appended
        assert n_calls == 0
