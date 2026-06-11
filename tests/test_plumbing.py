# -*- coding: utf-8 -

"""Testing the class NonConvex.

SPDX-FileCopyrightText: Patrik Schönfeldt

SPDX-License-Identifier: MIT
"""

import numpy as np
import pandas as pd
import pytest

from oemof.solph._plumbing import _FakeSequence
from oemof.solph._plumbing import sequence
from oemof.solph._plumbing import valid_sequence


def test_fake_sequence():
    seq_lt = _FakeSequence(7)
    seq_gt = _FakeSequence(64)
    seq0 = _FakeSequence(42.0)
    seq_mul = 2 * _FakeSequence(21.0)
    seq_rmul = _FakeSequence(2.0) * 21
    seq_div = _FakeSequence(84.0) / 2

    for seq in [seq0, seq_mul, seq_rmul, seq_div]:
        assert seq[0] == 42
        assert seq.size is None
        assert seq[10] == 42
        assert seq.size is None

        assert seq.max() == 42
        assert seq.min() == 42
        assert seq.value == 42
        assert seq.sum() == np.inf

        assert len(seq[0:1]) == 1
        assert len(seq[:2]) == 2
        assert len(seq[1:2]) == 1
        assert len(seq[1:-1]) == 0

        with pytest.raises(ValueError, match="The truth value of an array"):
            bool(seq_lt < seq)
        with pytest.raises(ValueError, match="The truth value of an array"):
            bool(seq_lt <= seq)
        with pytest.raises(ValueError, match="The truth value of an array"):
            bool(seq0 <= seq)
        with pytest.raises(ValueError, match="The truth value of an array"):
            bool(seq0 == seq)
        with pytest.raises(ValueError, match="The truth value of an array"):
            bool(seq0 >= seq)
        with pytest.raises(ValueError, match="The truth value of an array"):
            bool(seq_gt >= seq)
        with pytest.raises(ValueError, match="The truth value of an array"):
            bool(seq_gt > seq)

        for other in [42, [42, 42], np.array([42, 42])]:
            assert (seq_lt < other).all()
            assert (seq_lt <= other).all()
            assert (seq0 <= other).all()
            assert (seq0 == other).all()
            assert (seq0 >= other).all()
            assert (seq_gt >= other).all()
            assert (seq_gt > other).all()

            assert (seq_lt < other).any()
            assert (seq_lt <= other).any()
            assert (seq0 <= other).any()
            assert (seq0 == other).any()
            assert (seq0 >= other).any()
            assert (seq_gt >= other).any()
            assert (seq_gt > other).any()

        for other in [[1, 42, 100], np.array([1, 42, 100])]:
            assert not (seq_lt < other).all()
            assert not (seq_lt <= other).all()
            assert not (seq0 <= other).all()
            assert not (seq0 == other).all()
            assert not (seq0 >= other).all()
            assert not (seq_gt >= other).all()
            assert not (seq_gt > other).all()

            assert (seq_lt < other).any()
            assert (seq_lt <= other).any()
            assert (seq0 <= other).any()
            assert (seq0 == other).any()
            assert (seq0 >= other).any()
            assert (seq_gt >= other).any()
            assert (seq_gt > other).any()

        assert str(seq) == "[42.0, 42.0, ..., 42.0]"

        with pytest.raises(ValueError, match="Length needs to be defined"):
            seq.to_numpy()
        assert (seq.to_numpy(length=5) == np.array(5 * [42])).all()

        assert len(seq) == 0

        seq.size = 2
        assert seq.size == 2
        assert len(seq) == 2

        assert seq.max() == 42
        assert seq.min() == 42
        assert seq.value == 42
        assert seq.sum() == 84

        for other in [42, [42, 42]]:
            assert (other > seq_lt).all()
            assert (other >= seq_lt).all()
            assert (other >= seq0).all()
            assert (other == seq0).all()
            assert (other <= seq0).all()
            assert (other <= seq_gt).all()
            assert (other < seq_gt).all()

            assert (other > seq_lt).any()
            assert (other >= seq_lt).any()
            assert (other >= seq0).any()
            assert (other == seq0).any()
            assert (other <= seq0).any()
            assert (other <= seq_gt).any()
            assert (other < seq_gt).any()

        other = np.array([42, 42])
        # comparision from numpy only works with set size
        seq_lt.size = 2
        seq_gt.size = 2
        seq0.size = 2
        assert (other > seq_lt).all()
        assert (other >= seq_lt).all()
        assert (other >= seq0).all()
        assert (other == seq0).all()
        assert (other <= seq0).all()
        assert (other <= seq_gt).all()
        assert (other < seq_gt).all()

        assert (other > seq_lt).any()
        assert (other >= seq_lt).any()
        assert (other >= seq0).any()
        assert (other == seq0).any()
        assert (other <= seq0).any()
        assert (other <= seq_gt).any()
        assert (other < seq_gt).any()

        assert str(seq) == "[42.0, 42.0]"

        assert (seq.to_numpy() == np.array(2 * [42])).all()
        assert (seq.to_numpy(length=5) == np.array(5 * [42])).all()

        seq2 = np.array([2, 2]) * seq
        assert (seq2 == np.array(2 * [84])).all()


def test_sequence():
    seq0 = sequence(0)
    assert isinstance(seq0, _FakeSequence)
    assert seq0.value == 0
    assert seq0.size is None

    with pytest.raises(ValueError, match="Length mismatch"):
        _ = sequence([1, 3], length=3)
    seq13 = sequence([1, 3])
    assert isinstance(seq13, np.ndarray)
    assert (seq13 == np.array([1, 3])).all()

    with pytest.raises(ValueError, match="Length mismatch"):
        _ = sequence("ab", length=3)
    seq_ab = sequence("ab")
    assert isinstance(seq_ab, str)
    assert seq_ab == "ab"


def test_dimension_is_too_high_to_create_a_sequence():
    df = pd.DataFrame({"epc": 5}, index=["a"])
    with pytest.raises(ValueError, match="Dimension too high"):
        sequence(df)
    n2 = [[4]]
    with pytest.raises(ValueError, match="Dimension too high"):
        sequence(n2)


def test_valid_sequence():
    np_array = np.array([0, 1, 2, 3, 4])
    assert valid_sequence(np_array, 5)

    with pytest.warns(FutureWarning, match="Sequence longer than needed"):
        valid_sequence(np_array, 4)

    # it's not that long
    with pytest.raises(ValueError):
        valid_sequence(np_array, 1337)

    fake_sequence = _FakeSequence(42)
    assert valid_sequence(fake_sequence, 5)
    assert len(fake_sequence) == 5

    # wil not automatically overwrite size
    assert not valid_sequence(fake_sequence, 1337)
    assert len(fake_sequence) == 5

    # manually overwriting length is still possible
    fake_sequence.size = 1337
    assert valid_sequence(fake_sequence, 1337)
    assert len(fake_sequence) == 1337

    # strings are no valid sequences
    assert not valid_sequence("abc", 3)
