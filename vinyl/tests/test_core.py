import math

import pytest

import vinyl

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

S = [0.1] * 200

CASES = [
    ("len", lambda: len(vinyl.Vinyl(48000, 0.5, 1.0, 0.0, 1).process(S)) == 200),
    ("no_wow", lambda: all(abs(a - b) < 1e-3 for a, b in zip(vinyl.Vinyl(48000, 0.5, 0.0, 0.0, 1).process(S), S))),
    ("bad_depth", lambda: _raises(lambda: vinyl.Vinyl(48000, 0.5, -1.0, 0.0, 1))),
    ("bad_hz", lambda: _raises(lambda: vinyl.Vinyl(48000, 0.0, 1.0, 0.0, 1))),
    ("bad_noise", lambda: _raises(lambda: vinyl.Vinyl(48000, 0.5, 1.0, -0.1, 1))),
    ("det", lambda: vinyl.Vinyl(48000, 0.5, 1.0, 0.1, 1).process(S) == vinyl.Vinyl(48000, 0.5, 1.0, 0.1, 1).process(S)),
    ("seed_diff", lambda: vinyl.Vinyl(48000, 0.5, 1.0, 0.1, 1).process(S) != vinyl.Vinyl(48000, 0.5, 1.0, 0.1, 2).process(S)),
    ("empty", lambda: vinyl.Vinyl(48000, 0.5, 1.0, 0.0, 1).process([]) == []),
    ("type", lambda: isinstance(vinyl.Vinyl(48000, 0.5, 1.0, 0.0, 1).process(S), list)),
    ("range", lambda: all(abs(v) <= 1.0 for v in vinyl.Vinyl(48000, 0.5, 1.0, 0.0, 1).process(S))),
    ("nan", lambda: _raises(lambda: vinyl.Vinyl(48000, 0.5, 1.0, 0.0, 1).process([float('nan')]))),
    ("no_mut", lambda: vinyl.Vinyl(48000, 0.5, 1.0, 0.0, 1).process(list(S)) is not None),
    ("float", lambda: isinstance(vinyl.Vinyl(48000, 0.5, 1.0, 0.0, 1).process(S)[0], float)),
    ("wow_effect", lambda: vinyl.Vinyl(48000, 0.5, 1.0, 0.0, 1).process(S) != vinyl.Vinyl(48000, 0.5, 0.0, 0.0, 1).process(S)),
    ("noise_effect", lambda: vinyl.Vinyl(48000, 0.5, 0.0, 0.1, 1).process(S) != vinyl.Vinyl(48000, 0.5, 0.0, 0.0, 1).process(S)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
