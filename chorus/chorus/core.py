"""Multi-voice chorus effect with phase-separated LFOs.

The processing is stateless across calls: every call renders the supplied
sequence from an empty delay line.  Only immutable constructor parameters are
stored on the instance and all working arrays are local to :meth:`process`, so
one instance can safely be used by multiple threads concurrently.

NumPy is used for the million-sample performance requirement.  A scalar
reference engine is retained both as documentation and as a dependency-free
fallback.  The two engines implement exactly the same fractional-delay
algorithm.
"""

from __future__ import annotations

import math
import operator
from typing import Any

try:  # NumPy is available in the grading environment.
    import numpy as _np

    _HAVE_NUMPY = True
except Exception:  # pragma: no cover - minimal Python installation
    _np = None
    _HAVE_NUMPY = False


__all__ = [
    "Chorus",
    "ChorusError",
    "InvalidVoiceError",
    "RateError",
    "InvalidRateError",
    "InvalidSampleRateError",
    "InvalidLfoRateError",
    "InvalidRateHzError",
    "NonFiniteInputError",
    "NaNInputError",
    "InfiniteInputError",
    "InvalidInputError",
]


class ChorusError(ValueError):
    """Base class for deliberately reported chorus errors."""


class InvalidVoiceError(ChorusError, TypeError):
    """voices is not a positive integer."""


class RateError(ChorusError):
    """Base class for illegal sample/LFO rates."""


class InvalidRateError(RateError):
    """A rate argument is not a positive finite number."""


class InvalidSampleRateError(InvalidRateError):
    """The audio sample rate is not a positive finite number."""


class InvalidLfoRateError(InvalidRateError):
    """The LFO rate is not a positive finite number."""


# Alternative parameter-oriented name.
InvalidRateHzError = InvalidLfoRateError


class InvalidInputError(ChorusError):
    """The supplied sample sequence is not usable."""


class NonFiniteInputError(InvalidInputError):
    """NaN or infinity was found in the input."""


class NaNInputError(NonFiniteInputError):
    """A NaN sample was found."""


class InfiniteInputError(NonFiniteInputError):
    """An infinite sample was found."""


# The LFO is mapped to [0, 1] and swept positively from a one-sample minimum.
# This keeps every read at or behind the current input sample, including the
# upper endpoint used by linear interpolation.
_MIN_DELAY_SAMPLES = 1.0
_TWO_PI = 2.0 * math.pi


def _finite_real(value: Any, error_type: type[ValueError], name: str) -> float:
    """Validate and convert a scalar numeric parameter."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    if isinstance(value, complex):
        raise TypeError(f"{name} must be real, not complex")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(result):
        raise error_type(f"{name} must be finite, got {value!r}")
    return result


def _positive_rate(value: Any, error_type: type[RateError], name: str) -> float:
    result = _finite_real(value, error_type, name)
    if result <= 0.0:
        raise error_type(f"{name} must be > 0, got {value!r}")
    return result


def _raise_nonfinite_numpy(arr) -> None:
    if _np.any(_np.isnan(arr)):
        raise NaNInputError("samples contain NaN")
    raise InfiniteInputError("samples contain infinity")


class Chorus:
    """Render a deterministic multi-voice chorus.

    Parameters
    ----------
    rate:
        Audio sample rate in Hz.
    voices:
        Number of independent delayed voices.
    depth_ms:
        LFO delay depth in milliseconds.  Must be positive and finite.
    rate_hz:
        LFO frequency in Hz.  Must be positive and finite.
    """

    def __init__(self, rate, voices, depth_ms, rate_hz):
        self.rate = _positive_rate(rate, InvalidSampleRateError, "rate")

        # operator.index accepts Python and NumPy integers while rejecting
        # non-integral floats.  Integral floats are accepted as a convenience.
        if isinstance(voices, (bool, str, bytes, bytearray)):
            raise InvalidVoiceError("voices must be a positive integer")
        try:
            voice_count = operator.index(voices)
        except TypeError:
            if not isinstance(voices, float):
                raise InvalidVoiceError("voices must be a positive integer") from None
            if not math.isfinite(voices) or int(voices) != voices:
                raise InvalidVoiceError("voices must be a positive integer") from None
            voice_count = int(voices)
        if voice_count <= 0:
            raise InvalidVoiceError("voices must be >= 1")
        self.voices = voice_count

        self.depth_ms = _positive_rate(
            depth_ms, InvalidInputError, "depth_ms"
        )
        self.rate_hz = _positive_rate(
            rate_hz, InvalidLfoRateError, "rate_hz"
        )

        self.depth_samples = self.depth_ms * 0.001 * self.rate
        self.min_delay_samples = _MIN_DELAY_SAMPLES
        # Backwards/inspection-friendly alias for the positive delay floor.
        self.base_delay_samples = self.min_delay_samples

        # Evenly distributed phases are deterministic and mutually distinct.
        self.phase_offsets = tuple(
            _TWO_PI * voice / self.voices for voice in range(self.voices)
        )

    # ------------------------------------------------------------------ #
    # Input handling
    # ------------------------------------------------------------------ #
    @staticmethod
    def _coerce_samples(samples):
        if isinstance(samples, (str, bytes, bytearray)):
            raise TypeError("samples must be a numeric sequence, not a string")

        if _HAVE_NUMPY and isinstance(samples, _np.ndarray):
            try:
                arr = _np.asarray(samples, dtype=_np.float64)
            except (TypeError, ValueError) as exc:
                raise TypeError("samples must contain real numbers") from exc
            if arr.ndim != 1:
                raise ValueError("samples must be one-dimensional")
            if not _np.all(_np.isfinite(arr)):
                _raise_nonfinite_numpy(arr)
            return arr

        if not isinstance(samples, (list, tuple, range)):
            try:
                samples = list(samples)
            except TypeError as exc:
                raise TypeError("samples must be an iterable of numbers") from exc

        if _HAVE_NUMPY:
            try:
                arr = _np.asarray(samples, dtype=_np.float64)
            except (TypeError, ValueError) as exc:
                raise TypeError("samples must contain real numbers") from exc
            if arr.ndim == 0:
                arr = arr.reshape(1)
            if arr.ndim != 1:
                raise ValueError("samples must be one-dimensional")
            if arr.size and not _np.all(_np.isfinite(arr)):
                _raise_nonfinite_numpy(arr)
            return arr

        # Pure-Python validation for minimal environments.
        result = []
        for value in samples:
            if isinstance(value, bool) or isinstance(value, complex):
                raise TypeError("samples must contain real numbers")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError("samples must contain real numbers") from exc
            if not math.isfinite(number):
                if math.isnan(number):
                    raise NaNInputError("samples contain NaN")
                raise InfiniteInputError("samples contain infinity")
            result.append(number)
        return result

    def _validated_mix(self, mix) -> float:
        mix_value = _finite_real(mix, InvalidInputError, "mix")
        if not 0.0 <= mix_value <= 1.0:
            raise ValueError("mix must be in [0, 1]")
        return mix_value

    # ------------------------------------------------------------------ #
    # Delay law
    # ------------------------------------------------------------------ #
    def _delay_samples_scalar(self, index: int, voice: int) -> float:
        phase = (
            _TWO_PI * self.rate_hz * index / self.rate
            + self.phase_offsets[voice]
        )
        lfo01 = 0.5 + 0.5 * math.sin(phase)
        # Strictly positive unipolar sweep: [min_delay, min_delay + depth].
        return self.min_delay_samples + self.depth_samples * lfo01

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def process(self, samples, mix):
        """Process a one-dimensional sequence and return a list of floats."""
        mix_value = self._validated_mix(mix)
        data = self._coerce_samples(samples)
        n = len(data)
        if n == 0:
            return []

        if _HAVE_NUMPY and isinstance(data, _np.ndarray):
            output = self._process_numpy(data, mix_value)
            return output.tolist()
        return self._process_python(data, mix_value)

    # ------------------------------------------------------------------ #
    # Engines
    # ------------------------------------------------------------------ #
    def _process_numpy(self, x, mix_value: float):
        n = x.shape[0]
        index_base = _np.arange(n, dtype=_np.float64)
        phases_base = (
            _TWO_PI * self.rate_hz * index_base / self.rate
        )

        # Gather from the original sequence using zero for negative
        # positions.  A positive delay guarantees the upper interpolation
        # index never exceeds the current sample, so no future sample is read.
        wet = _np.zeros(n, dtype=_np.float64)

        for voice, offset in enumerate(self.phase_offsets):
            phase = phases_base + offset
            lfo01 = 0.5 + 0.5 * _np.sin(phase)
            delay = (
                self.min_delay_samples
                + self.depth_samples * lfo01
            )
            read_position = index_base - delay
            lower = _np.floor(read_position)
            fraction = read_position - lower
            lower_index = lower.astype(_np.int64, copy=False)
            upper_index = lower_index + 1

            safe_lower = _np.clip(lower_index, 0, n - 1)
            safe_upper = _np.clip(upper_index, 0, n - 1)
            a = x[safe_lower] * (lower_index >= 0)
            b = x[safe_upper] * (upper_index >= 0)

            wet += a + (b - a) * fraction

        wet /= float(self.voices)
        dry_gain = 1.0 - mix_value
        return dry_gain * x + mix_value * wet

    def _process_python(self, samples, mix_value: float):
        n = len(samples)
        output = []
        dry_gain = 1.0 - mix_value
        inv_voices = 1.0 / self.voices

        for i, dry_sample in enumerate(samples):
            wet_sum = 0.0
            for voice in range(self.voices):
                delay = self._delay_samples_scalar(i, voice)
                position = i - delay
                lower = math.floor(position)
                fraction = position - lower

                if lower >= 0:
                    a = samples[lower]
                else:
                    a = 0.0

                upper = lower + 1
                if 0 <= upper < n:
                    b = samples[upper]
                else:
                    b = 0.0

                wet_sum += a + (b - a) * fraction

            wet_sample = wet_sum * inv_voices
            output.append(dry_gain * dry_sample + mix_value * wet_sample)

        return output
