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
import math

import pytest

from trl.experimental.async_grpo.async_rollout_worker import AsyncRolloutLoop, RolloutGroup, TrainingSequence

from ..testing_utils import TrlTestCase


@pytest.fixture(scope="module", autouse=True)
def _init_accelerate_state():
    # _score_group logs via accelerate's logger, which requires PartialState() to be initialized
    # (the real worker does this in _child_main before running the loop).
    from accelerate.state import PartialState

    PartialState()


def two_reward(completions, **kwargs):
    # Deterministic per-conversation reward: two conversations get rewards 1.0 and 3.0.
    return [1.0, 3.0]


def _bare_loop(reward_funcs):
    # _score_group only reads reward_funcs / reward_func_names off self, so we skip the heavy __init__
    # (tokenizer, asyncio loop, environments) and set just those two attributes.
    loop = object.__new__(AsyncRolloutLoop)
    loop.reward_funcs = reward_funcs
    loop.reward_func_names = [f.__name__ for f in reward_funcs]
    return loop


def _group(completions_sequences, completions_ids):
    n = len(completions_sequences)
    return RolloutGroup(
        prompt=[{"role": "user", "content": "hi"}],
        prompt_ids=[1, 2, 3],
        reward_kwargs={},
        completions=[[{"role": "assistant", "content": f"c{i}"}] for i in range(n)],
        completions_ids=completions_ids,
        completions_sequences=completions_sequences,
        tool_call_counts=[0] * n,
        tool_failure_counts=[0] * n,
        model_version=7,
    )


class TestScoreGroupOptionThree(TrlTestCase):
    def test_one_advantage_per_conversation_stamped_on_every_row(self):
        # conv 0: one row; conv 1: two rows (a fork).
        seq_a = TrainingSequence([1, 2, 3, 10, 11], [0, 0, 0, 1, 1], [0, 0, 0, -0.1, -0.2], "c0")
        seq_b1 = TrainingSequence([1, 2, 3, 20, 21], [0, 0, 0, 1, 1], [0, 0, 0, -0.3, -0.4], "c1")
        seq_b2 = TrainingSequence([1, 2, 3, 20, 99, 30], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, -0.5], "c1")
        group = _group([[seq_a], [seq_b1, seq_b2]], completions_ids=[[10, 11], [20, 21, 30]])

        samples = asyncio.run(_bare_loop([two_reward])._score_group(group))

        # 1 row for conv 0 + 2 rows for conv 1 = 3 samples.
        assert len(samples) == 3

        # rewards [1, 3] -> mean 2, std 1 -> advantages [-1, +1].
        assert samples[0].advantage == pytest.approx(-1.0)
        assert samples[1].advantage == pytest.approx(1.0)
        assert samples[2].advantage == pytest.approx(1.0)
        # The fork's two rows carry the identical (conversation-level) advantage — no split.
        assert samples[1].advantage == samples[2].advantage

        # Each sample maps 1:1 onto its source TrainingSequence.
        assert samples[0].input_ids == seq_a.input_ids and samples[0].completion_mask == seq_a.completion_mask
        assert samples[0].old_log_probs == seq_a.old_log_probs
        assert samples[1].input_ids == seq_b1.input_ids
        assert samples[2].input_ids == seq_b2.input_ids

        assert all(s.model_version == 7 for s in samples)

    def test_metrics_are_per_conversation_and_independent(self):
        seq_a = TrainingSequence([1, 2, 10], [0, 0, 1], [0, 0, -0.1], "c0")
        seq_b1 = TrainingSequence([1, 2, 20], [0, 0, 1], [0, 0, -0.2], "c1")
        seq_b2 = TrainingSequence([1, 2, 20, 30], [0, 0, 0, 1], [0, 0, 0, -0.3], "c1")
        group = _group([[seq_a], [seq_b1, seq_b2]], completions_ids=[[10], [20, 30]])

        samples = asyncio.run(_bare_loop([two_reward])._score_group(group))

        assert samples[0].metrics["reward"] == 1.0
        assert samples[1].metrics["reward"] == 3.0 and samples[2].metrics["reward"] == 3.0
        assert samples[0].metrics["reward_std"] == pytest.approx(1.0)
        assert samples[1].metrics["rewards/two_reward"] == 3.0
        # The fork's two rows must not share a metrics dict (the score loop mutates it per sample).
        assert samples[1].metrics is not samples[2].metrics

    def test_all_none_reward_conversation_is_unscorable(self):
        # A conversation for which every reward func returns None gets advantage 0 and NaN reward.
        def maybe_none(completions, **kwargs):
            return [None, 2.0]

        seq_a = TrainingSequence([1, 2, 10], [0, 0, 1], [0, 0, -0.1], "c0")
        seq_b = TrainingSequence([1, 2, 20], [0, 0, 1], [0, 0, -0.2], "c1")
        group = _group([[seq_a], [seq_b]], completions_ids=[[10], [20]])

        samples = asyncio.run(_bare_loop([maybe_none])._score_group(group))

        assert len(samples) == 2
        assert samples[0].advantage == 0.0  # unscorable -> advantage 0
        assert math.isnan(samples[0].metrics["reward"])
        assert samples[1].advantage == 0.0  # only one scorable row -> zero-centered
        assert samples[1].metrics["reward"] == 2.0
