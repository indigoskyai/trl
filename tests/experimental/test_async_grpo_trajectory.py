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

from trl.experimental.async_grpo.async_rollout_worker import (
    DriftKind,
    TurnRecord,
    _chain_to_sequences,
    _common_prefix_len,
    _SampleBuilder,
)

from ..testing_utils import TrlTestCase


def _finalize(turns, rollout_id="r0", fork_threshold=1024):
    return _chain_to_sequences(turns, rollout_id, fork_threshold)


class TestReconciler(TrlTestCase):
    def test_common_prefix_len(self):
        assert _common_prefix_len([1, 2, 3], [1, 2, 3]) == 3  # identical
        assert _common_prefix_len([1, 2], [1, 2, 3, 4]) == 2  # old is a prefix of new
        assert _common_prefix_len([], [1, 2, 3]) == 0  # empty
        # chunk=2 exercises both the whole-chunk fast path and the per-token fallback
        assert _common_prefix_len([1, 2, 3, 4, 5], [1, 2, 3, 9, 5], chunk=2) == 3
        assert _common_prefix_len([1, 2, 3, 4], [1, 2, 3, 4, 5], chunk=2) == 4

    def test_single_turn(self):
        rows = _finalize([TurnRecord([1, 2, 3], [10, 11], [-0.1, -0.2])])
        assert len(rows) == 1
        assert rows[0].input_ids == [1, 2, 3, 10, 11]
        assert rows[0].completion_mask == [0, 0, 0, 1, 1]
        assert rows[0].old_log_probs == [0.0, 0.0, 0.0, -0.1, -0.2]
        assert rows[0].rollout_id == "r0"

    def test_clean_chain_stays_one_row(self):
        # Turn 2's re-tokenized prompt starts with the held tokens (gen [10,11] + tool [20,21]) -> CLEAN.
        turn1 = TurnRecord([1, 2, 3], [10, 11], [-0.1, -0.2])
        turn2 = TurnRecord([1, 2, 3, 10, 11, 20, 21], [30, 31], [-0.3, -0.4])
        rows = _finalize([turn1, turn2])
        assert len(rows) == 1
        assert rows[0].input_ids == [1, 2, 3, 10, 11, 20, 21, 30, 31]
        assert rows[0].completion_mask == [0, 0, 0, 1, 1, 0, 0, 1, 1]  # prompt=0 gen=1 tool=0 gen=1
        assert rows[0].old_log_probs == [0.0, 0.0, 0.0, -0.1, -0.2, 0.0, 0.0, -0.3, -0.4]

    def test_rewrite_forks_into_two_rows(self):
        # Divergence inside turn 1's answer + a turn >= fork_threshold -> FORK. Every generated token is
        # trained in exactly one row (turn 1's tokens are context in row 2).
        turn1 = TurnRecord([1, 2, 3], [10, 11, 12, 13])
        turn2 = TurnRecord([1, 2, 3, 10, 99, 88, 77], [30, 31, 32])
        rows = _finalize([turn1, turn2], fork_threshold=2)
        assert len(rows) == 2
        assert rows[0].input_ids == [1, 2, 3, 10, 11, 12, 13]
        assert rows[0].completion_mask == [0, 0, 0, 1, 1, 1, 1]
        assert rows[1].input_ids == [1, 2, 3, 10, 99, 88, 77, 30, 31, 32]
        assert rows[1].completion_mask == [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]

    def test_fork_when_divergence_precedes_last_response(self):
        # matched < last_response_start_idx -> FORK regardless of threshold (distinct from the length trigger).
        builder = _SampleBuilder(fork_threshold=1024)
        builder.append_turn(TurnRecord([1, 2, 3], [10, 11]), DriftKind.CLEAN)  # last_response_start_idx == 3
        assert builder.classify_token_drift(TurnRecord([1, 9, 3, 10, 11], [30])) is DriftKind.FORK

    def test_tail_wobble_realigns_to_context(self):
        # Last generated token re-renders (11 -> 12): a short wobble in the last answer -> REALIGN. The
        # re-rendered tail becomes context, so turn 1 loses its training signal.
        turn1 = TurnRecord([1, 2, 3], [10, 11], [-0.1, -0.2])
        turn2 = TurnRecord([1, 2, 3, 10, 12], [30, 31], [-0.3, -0.4])
        rows = _finalize([turn1, turn2])
        assert len(rows) == 1
        assert rows[0].input_ids == [1, 2, 3, 10, 12, 30, 31]
        assert rows[0].completion_mask == [0, 0, 0, 0, 0, 1, 1]
        assert rows[0].old_log_probs == [0.0, 0.0, 0.0, 0.0, 0.0, -0.3, -0.4]
