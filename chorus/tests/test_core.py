import math

import pytest

import chorus

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

S = [0.1] * 100

CASES = [
    ("len", lambda: len(chorus.Chorus(48000, 3, 5.0, 0.5).process(S, 0.5)) == 100),
    ("dry", lambda: abs(chorus.Chorus(48000, 3, 5.0, 0.5).process(S, 0.0)[0] - 0.1) < 1e-9),
    ("bad_voices", lambda: _raises(lambda: chorus.Chorus(48000, 0, 5.0, 0.5))),
    ("bad_depth", lambda: _raises(lambda: chorus.Chorus(48000, 3, 0.0, 0.5))),
    ("bad_rate", lambda: _raises(lambda: chorus.Chorus(48000, 3, 5.0, 0.0))),
    ("bad_mix", lambda: _raises(lambda: chorus.Chorus(48000, 3, 5.0, 0.5).process(S, 1.5))),
    ("empty", lambda: chorus.Chorus(48000, 3, 5.0, 0.5).process([], 0.5) == []),
    ("type", lambda: isinstance(chorus.Chorus(48000, 3, 5.0, 0.5).process(S, 0.5), list)),
    ("det", lambda: chorus.Chorus(48000, 3, 5.0, 0.5).process(S, 0.5) == chorus.Chorus(48000, 3, 5.0, 0.5).process(S, 0.5)),
    ("range", lambda: all(abs(v) <= 1.0 for v in chorus.Chorus(48000, 3, 5.0, 0.5).process(S, 0.5))),
    ("nan", lambda: _raises(lambda: chorus.Chorus(48000, 3, 5.0, 0.5).process([float('nan')], 0.5))),
    ("no_mut", lambda: chorus.Chorus(48000, 3, 5.0, 0.5).process(list(S), 0.5) is not None),
    ("wet_differs", lambda: chorus.Chorus(48000, 3, 5.0, 0.5).process(S, 1.0) != chorus.Chorus(48000, 3, 5.0, 0.5).process(S, 0.0)),
    ("float", lambda: isinstance(chorus.Chorus(48000, 3, 5.0, 0.5).process(S, 0.5)[0], float)),
    ("bad_sr", lambda: _raises(lambda: chorus.Chorus(0, 3, 5.0, 0.5))),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
