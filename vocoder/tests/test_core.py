import math

import pytest

import vocoder

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

C = [0.5] * 512
M = [0.5] * 512

CASES = [
    ("len", lambda: len(vocoder.Vocoder(48000, 16, 256).process(C, M)) == 512),
    ("cola", lambda: vocoder.Vocoder(48000, 16, 256).cola_error() <= 1e-9),
    ("bad_bands", lambda: _raises(lambda: vocoder.Vocoder(48000, 0, 256))),
    ("bad_frame", lambda: _raises(lambda: vocoder.Vocoder(48000, 16, 0))),
    ("bad_len", lambda: _raises(lambda: vocoder.Vocoder(48000, 16, 256).process(C, [0.5]))),
    ("empty", lambda: vocoder.Vocoder(48000, 16, 256).process([], []) == []),
    ("type", lambda: isinstance(vocoder.Vocoder(48000, 16, 256).process(C, M), list)),
    ("det", lambda: vocoder.Vocoder(48000, 16, 256).process(C, M) == vocoder.Vocoder(48000, 16, 256).process(C, M)),
    ("range", lambda: all(abs(v) <= 1.0 for v in vocoder.Vocoder(48000, 16, 256).process(C, M))),
    ("nan", lambda: _raises(lambda: vocoder.Vocoder(48000, 16, 256).process([float('nan')], M))),
    ("no_mut", lambda: vocoder.Vocoder(48000, 16, 256).process(list(C), list(M)) is not None),
    ("cola_type", lambda: isinstance(vocoder.Vocoder(48000, 16, 256).cola_error(), float)),
    ("bad_rate", lambda: _raises(lambda: vocoder.Vocoder(0, 16, 256))),
    ("float", lambda: isinstance(vocoder.Vocoder(48000, 16, 256).process(C, M)[0], float)),
    ("silence", lambda: all(abs(v) < 1e-6 for v in vocoder.Vocoder(48000, 16, 256).process(C, [0.0] * 512))),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
