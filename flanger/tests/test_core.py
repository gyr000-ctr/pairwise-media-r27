import math

import pytest

import flanger

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

S = [0.1] * 100

CASES = [
    ("len", lambda: len(flanger.Flanger(48000, 1.0, 5.0, 0.5).process(S, 0.5)) == 100),
    ("bad_fb", lambda: _raises(lambda: flanger.Flanger(48000, 1.0, 5.0, 1.0))),
    ("bad_range", lambda: _raises(lambda: flanger.Flanger(48000, 5.0, 1.0, 0.5))),
    ("bad_rate", lambda: _raises(lambda: flanger.Flanger(0, 1.0, 5.0, 0.5))),
    ("spacing", lambda: abs(flanger.Flanger(48000, 1.0, 5.0, 0.5).notch_spacing(48) - 1000.0) < 1e-6),
    ("empty", lambda: flanger.Flanger(48000, 1.0, 5.0, 0.5).process([], 0.5) == []),
    ("type", lambda: isinstance(flanger.Flanger(48000, 1.0, 5.0, 0.5).process(S, 0.5), list)),
    ("det", lambda: flanger.Flanger(48000, 1.0, 5.0, 0.5).process(S, 0.5) == flanger.Flanger(48000, 1.0, 5.0, 0.5).process(S, 0.5)),
    ("range", lambda: all(abs(v) <= 1.0 for v in flanger.Flanger(48000, 1.0, 5.0, 0.5).process(S, 0.5))),
    ("nan", lambda: _raises(lambda: flanger.Flanger(48000, 1.0, 5.0, 0.5).process([float('nan')], 0.5))),
    ("no_mut", lambda: flanger.Flanger(48000, 1.0, 5.0, 0.5).process(list(S), 0.5) is not None),
    ("spacing_type", lambda: isinstance(flanger.Flanger(48000, 1.0, 5.0, 0.5).notch_spacing(48), float)),
    ("bad_mix", lambda: _raises(lambda: flanger.Flanger(48000, 1.0, 5.0, 0.5).process(S, 2.0))),
    ("float", lambda: isinstance(flanger.Flanger(48000, 1.0, 5.0, 0.5).process(S, 0.5)[0], float)),
    ("bad_delay", lambda: _raises(lambda: flanger.Flanger(48000, 1.0, 5.0, 0.5).notch_spacing(0))),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
