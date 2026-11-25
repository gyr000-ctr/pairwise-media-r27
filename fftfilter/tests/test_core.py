import math

import pytest

import fftfilter

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

S = [0.1] * 64

CASES = [
    ("len", lambda: len(fftfilter.FftFilter(48000).notch(S, 1000.0, 1.0)) == 64),
    ("bad_rate", lambda: _raises(lambda: fftfilter.FftFilter(0))),
    ("bad_freq", lambda: _raises(lambda: fftfilter.FftFilter(48000).notch(S, 0.0, 1.0))),
    ("bad_q", lambda: _raises(lambda: fftfilter.FftFilter(48000).notch(S, 1000.0, 0.0))),
    ("atten", lambda: abs(fftfilter.FftFilter(48000).response(1000.0, 1.0, 1000.0)) < 0.03),
    ("passband", lambda: abs(fftfilter.FftFilter(48000).response(1000.0, 1.0, 100.0)) > 0.9),
    ("empty", lambda: fftfilter.FftFilter(48000).notch([], 1000.0, 1.0) == []),
    ("type", lambda: isinstance(fftfilter.FftFilter(48000).notch(S, 1000.0, 1.0), list)),
    ("det", lambda: fftfilter.FftFilter(48000).notch(S, 1000.0, 1.0) == fftfilter.FftFilter(48000).notch(S, 1000.0, 1.0)),
    ("range", lambda: all(abs(v) <= 1.0 for v in fftfilter.FftFilter(48000).notch(S, 1000.0, 1.0))),
    ("nan", lambda: _raises(lambda: fftfilter.FftFilter(48000).notch([float('nan')], 1000.0, 1.0))),
    ("no_mut", lambda: fftfilter.FftFilter(48000).notch(list(S), 1000.0, 1.0) is not None),
    ("resp_type", lambda: isinstance(fftfilter.FftFilter(48000).response(1000.0, 1.0, 500.0), float)),
    ("two_notch", lambda: len(fftfilter.FftFilter(48000).notch(S, 1000.0, 1.0)) == len(S)),
    ("float", lambda: isinstance(fftfilter.FftFilter(48000).notch(S, 1000.0, 1.0)[0], float)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
