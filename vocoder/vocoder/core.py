"""Channel vocoder.

The vocoder analyses the *modulator* into a bank of logarithmically spaced,
slightly overlapping bands, smooths each band envelope with independent
attack / release time constants and applies the resulting per-band gains to the
*carrier*, which is re-synthesised with a COLA (constant-overlap-add) window.

Only NumPy is used for the heavy lifting.  ``process`` keeps all of its state
local, so a single :class:`Vocoder` instance is safe to share between threads.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "Vocoder",
    "VocoderError",
    "InvalidSampleRate",
    "InvalidBandCount",
    "InvalidFrameSize",
    "InvalidEnvelopeTime",
    "InvalidParameter",
    "InvalidSampleType",
    "InvalidInputLength",
    "NonFiniteInput",
]


# ---------------------------------------------------------------------------
# Errors.  Every public error derives from ``ValueError`` so that callers that
# only care about "invalid argument" keep working, while the concrete classes
# and messages let the different failure modes be told apart.
# ---------------------------------------------------------------------------
class VocoderError(ValueError):
    """Base class for all vocoder argument/input errors."""


class InvalidSampleRate(VocoderError):
    """Sample rate must be a positive finite number."""


class InvalidBandCount(VocoderError):
    """Band count must be a positive integer that fits the frame resolution."""


class InvalidFrameSize(VocoderError):
    """Frame size must be a positive integer."""


class InvalidParameter(VocoderError):
    """A numeric configuration parameter is out of range."""


class InvalidEnvelopeTime(InvalidParameter):
    """Attack / release time constant is out of range."""


class InvalidSampleType(VocoderError):
    """An input sequence contains non numeric data."""


class InvalidInputLength(VocoderError):
    """Carrier and modulator lengths differ."""


class NonFiniteInput(VocoderError):
    """An input sequence contains NaN or infinity."""


# ---------------------------------------------------------------------------
# Small validation helpers
# ---------------------------------------------------------------------------
def _is_integral(value):
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, np.integer)):
        return True
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value)) and float(value).is_integer()
    return False


def _as_int(value):
    return int(value)


class Vocoder:
    """A channel vocoder.

    Parameters
    ----------
    rate:
        Sample rate in Hz (positive, finite).
    bands:
        Number of log-spaced channels (positive integer).
    frame:
        Analysis/synthesis window length in samples (positive integer).
    attack_ms, release_ms:
        Envelope follower time constants.  ``attack_ms`` < ``release_ms`` by
        default, giving the usual fast-attack / slow-release behaviour.  The
        follower runs across frames (one update per hop), so its state is
        continuous in time even when ``process`` is called on short buffers.
    """

    #: Highest gain ever applied to a single band.  Prevents a silent carrier
    #: from producing enormous -- and unstable -- amplification.
    MAX_GAIN = 1.0e3

    def __init__(
        self,
        rate,
        bands,
        frame,
        attack_ms=1.0,
        release_ms=10.0,
        *,
        attack=None,
        release=None,
    ):
        """Construct a vocoder; see the class docstring for the parameters."""
        if attack is not None:
            attack_ms = attack
        if release is not None:
            release_ms = release
        if isinstance(rate, bool) or not isinstance(rate, (int, float, np.integer, np.floating)):
            raise InvalidSampleRate("sample rate must be a real number")
        rate = float(rate)
        if not math.isfinite(rate) or rate <= 0.0:
            raise InvalidSampleRate(
                "sample rate must be a positive finite number, got %r" % (rate,)
            )

        if not _is_integral(bands):
            raise InvalidBandCount("band count must be an integer, got %r" % (bands,))
        bands = _as_int(bands)
        if bands <= 0:
            raise InvalidBandCount("band count must be >= 1, got %d" % bands)

        if not _is_integral(frame):
            raise InvalidFrameSize("frame size must be an integer, got %r" % (frame,))
        frame = _as_int(frame)
        if frame <= 0:
            raise InvalidFrameSize("frame size must be >= 1, got %d" % frame)

        # A band count larger than the frame itself (or so large that the
        # weight matrix would not fit in memory) cannot be represented.
        max_bands = max(1, frame)
        if bands > max_bands:
            raise InvalidBandCount(
                "band count %d exceeds the frame size %d" % (bands, frame)
            )
        bins = frame // 2 + 1
        if bands * bins > 8_000_000:
            raise InvalidBandCount(
                "band count %d is too large to resolve with a %d-sample frame"
                % (bands, frame)
            )

        for name, value in (("attack_ms", attack_ms), ("release_ms", release_ms)):
            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise InvalidEnvelopeTime("%s must be a real number" % name)
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise InvalidEnvelopeTime(
                    "%s must be a positive finite number, got %r" % (name, value)
                )
            if name == "attack_ms":
                attack = value
            else:
                release = value

        self.rate = rate
        self.bands = bands
        self.frame = frame
        self.hop = max(1, frame // 2)

        self.attack_ms = attack
        self.release_ms = release
        # Convenience aliases.
        self.attack = attack
        self.release = release

        # COLA window pair.  The analysis window itself satisfies COLA with the
        # chosen hop; the synthesis window is flat, so the product is COLA too.
        self._analysis_window, self._synthesis_window = self._design_windows(
            frame, self.hop
        )
        self._cola_error = self._measure_cola(
            self._analysis_window, self._synthesis_window, frame, self.hop
        )

        self._band_centers, self._band_edges, self._band_weights = self._design_bands()

        # Exponential attack / release coefficients, one per hop.
        hop_seconds = self.hop / rate
        self._attack_coef = math.exp(-hop_seconds / (attack * 1e-3))
        self._release_coef = math.exp(-hop_seconds / (release * 1e-3))

    # ------------------------------------------------------------------
    # Public read-only views (useful for introspection / tests)
    # ------------------------------------------------------------------
    @property
    def analysis_window(self):
        return list(self._analysis_window)

    @property
    def synthesis_window(self):
        return list(self._synthesis_window)

    @property
    def window(self):
        """Alias for :attr:`analysis_window`."""
        return self.analysis_window

    @property
    def band_centers(self):
        """Log-spaced band centre frequencies in Hz."""
        return list(self._band_centers)

    @property
    def center_frequencies(self):
        """Alias for :attr:`band_centers`."""
        return self.band_centers

    @property
    def band_edges(self):
        """``(low, high)`` frequency bounds in Hz for every band."""
        return [(float(lo), float(hi)) for lo, hi in self._band_edges]

    @property
    def filter_bands(self):
        """Alias for :attr:`band_edges`."""
        return self.band_edges

    @property
    def band_count(self):
        return self.bands

    @property
    def band_weights(self):
        """Normalised FFT-bin weights, shape ``(bands, frame // 2 + 1)``."""
        return [list(row) for row in self._band_weights]

    # ------------------------------------------------------------------
    # Window design
    # ------------------------------------------------------------------
    @staticmethod
    def _design_windows(frame, hop):
        """Build a complementary (COLA) analysis/synthesis window pair.

        The analysis window is a periodic Hann prototype whose value is scaled
        per residue class modulo the hop, so that the overlap-add of the window
        over consecutive frames sums to exactly one for every output sample::

            sum_k a[n - k hop] == 1

        This also holds for odd frame lengths, where the number of overlapping
        frames alternates between two values.  The synthesis window is flat, so
        the analysis/synthesis product is COLA as well::

            sum_k a[n - k hop] * s[n - k hop] == 1

        Reconstruction is therefore exact when the band gains are all one.
        """
        n = np.arange(frame, dtype=np.float64)
        if frame == 1:
            return np.ones(1), np.ones(1)
        if frame == 2:
            prototype = np.ones(2, dtype=np.float64)
        else:
            prototype = 0.5 - 0.5 * np.cos(2.0 * math.pi * n / frame)

        # For every residue r modulo the hop, the overlap-add of a window at an
        # interior output sample is the sum of that whole residue class.
        level = np.zeros(hop, dtype=np.float64)
        for r in range(hop):
            level[r] = float(np.sum(prototype[r::hop]))

        zero = level <= 0.0
        if np.any(zero):  # pragma: no cover - degenerate prototype only
            level = np.where(zero, 1.0, level)

        scale = 1.0 / level[np.arange(frame) % hop]
        analysis = prototype * scale
        synthesis = np.ones(frame, dtype=np.float64)
        return analysis, synthesis

    @staticmethod
    def _measure_cola(analysis, synthesis, frame, hop):
        """Max deviation of the overlap-add of a*s from one, per residue."""
        product = analysis * synthesis
        error = 0.0
        for r in range(hop):
            total = float(np.sum(product[r::hop]))
            error = max(error, abs(total - 1.0))
        return error

    # ------------------------------------------------------------------
    # Band design
    # ------------------------------------------------------------------
    def _design_bands(self):
        frame = self.frame
        rate = self.rate
        nyquist = rate * 0.5
        bin_hz = rate / float(frame)

        f_low = max(20.0, bin_hz * 0.5)
        f_high = nyquist
        if f_low >= f_high:
            f_low = max(1e-9, f_high * 1e-3)

        bands = self.bands
        if bands == 1:
            centers = np.array([math.sqrt(f_low * f_high)], dtype=np.float64)
            lows = np.array([0.0], dtype=np.float64)
            highs = np.array([f_high], dtype=np.float64)
        else:
            ratio = (f_high / f_low) ** (1.0 / (bands - 1))
            centers = f_low * ratio ** np.arange(bands, dtype=np.float64)
            half = ratio ** 0.75  # > sqrt(ratio): adjacent bands overlap
            lows = centers / half
            highs = centers * half
            # Extend the outer bands to cover DC and Nyquist, so that the band
            # weights form a partition of unity over the whole spectrum.
            lows[0] = 0.0
            highs[-1] = f_high

        freqs = np.fft.rfftfreq(frame, d=1.0 / rate)
        raw = ((freqs[None, :] >= lows[:, None]) & (freqs[None, :] <= highs[:, None]))
        raw = raw.astype(np.float64)

        column_sum = raw.sum(axis=0)
        missing = column_sum == 0.0
        if np.any(missing):
            # Numerical safety net: assign uncovered bins to the nearest centre.
            missing_idx = np.nonzero(missing)[0]
            for idx in missing_idx:
                nearest = int(np.argmin(np.abs(centers - freqs[idx])))
                raw[nearest, idx] = 1.0
            column_sum = raw.sum(axis=0)

        weights = raw / column_sum[None, :]
        edges = list(zip(lows.tolist(), highs.tolist()))
        return centers.tolist(), edges, weights

    # ------------------------------------------------------------------
    # Envelope follower
    # ------------------------------------------------------------------
    def _smooth_envelope(self, raw):
        """Attack/release smoothing along the frame axis, continuous in time.

        ``raw`` has shape ``(frames, channels)``.  Each channel has its own
        follower state, which is carried from frame to frame; the very first
        frame starts from silence.  A per-column pure-Python loop is markedly
        faster than a per-row NumPy loop here because the frame count dwarfs
        the band count.
        """
        frames, channels = raw.shape
        out = np.empty_like(raw)
        attack = self._attack_coef
        release = self._release_coef
        for channel in range(channels):
            xs = raw[:, channel].tolist()
            ys = [0.0] * frames
            state = 0.0
            for index in range(frames):
                target = xs[index]
                coef = attack if target > state else release
                state = coef * state + (1.0 - coef) * target
                ys[index] = state
            out[:, channel] = ys
        return out

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------
    @staticmethod
    def _to_array(values, name):
        try:
            array = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError):
            # Iterators / generators are not understood by ``asarray``.
            try:
                array = np.asarray(list(values), dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise InvalidSampleType(
                    "%s contains non numeric data" % name
                ) from exc
        if array.ndim != 1:
            raise InvalidSampleType("%s must be a one dimensional sequence" % name)
        if array.size and not np.all(np.isfinite(array)):
            raise NonFiniteInput("%s contains NaN or infinity" % name)
        return array

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------
    def process(self, carrier, modulator):
        """Vocode ``carrier`` with the spectral envelope of ``modulator``.

        Both sequences must have the same length.  A new list of ``float`` of
        that same length is returned; the inputs are never modified.
        """
        carrier = self._to_array(carrier, "carrier")
        modulator = self._to_array(modulator, "modulator")

        if carrier.size != modulator.size:
            raise InvalidInputLength(
                "carrier and modulator must have equal length (%d != %d)"
                % (carrier.size, modulator.size)
            )
        length = int(carrier.size)
        if length == 0:
            return []

        frame = self.frame
        hop = self.hop

        # Enough zero padding on both sides that every output sample receives
        # the full COLA contribution.
        padded_length = length + 2 * frame
        carrier_padded = np.zeros(padded_length, dtype=np.float64)
        modulator_padded = np.zeros(padded_length, dtype=np.float64)
        carrier_padded[frame : frame + length] = carrier
        modulator_padded[frame : frame + length] = modulator

        frame_count = (padded_length - frame) // hop + 1
        sample_index = np.arange(frame, dtype=np.int64)
        starts = np.arange(frame_count, dtype=np.int64) * hop
        index = starts[:, None] + sample_index[None, :]

        window = self._analysis_window
        carrier_frames = carrier_padded[index] * window[None, :]
        modulator_frames = modulator_padded[index] * window[None, :]

        carrier_spectrum = np.fft.rfft(carrier_frames, axis=1)
        modulator_spectrum = np.fft.rfft(modulator_frames, axis=1)

        weights = self._band_weights
        weights_t = weights.T

        carrier_power = carrier_spectrum.real ** 2 + carrier_spectrum.imag ** 2
        modulator_power = modulator_spectrum.real ** 2 + modulator_spectrum.imag ** 2

        carrier_envelope = np.sqrt(np.maximum(carrier_power @ weights_t, 0.0))
        modulator_envelope = np.sqrt(np.maximum(modulator_power @ weights_t, 0.0))

        # Smooth both envelopes in one pass (carrier bands first, then
        # modulator bands) so the stateful loop runs once, not twice.
        smoothed = self._smooth_envelope(
            np.concatenate((carrier_envelope, modulator_envelope), axis=1)
        )
        bands = self.bands
        carrier_envelope = smoothed[:, :bands]
        modulator_envelope = smoothed[:, bands:]

        # A silent carrier cannot be shaped; a silent modulator must produce
        # silence.  In both cases the gain is zero.
        epsilon = 1.0e-15
        gain = np.zeros_like(modulator_envelope)
        np.divide(
            modulator_envelope,
            carrier_envelope,
            out=gain,
            where=carrier_envelope > epsilon,
        )
        np.clip(gain, 0.0, self.MAX_GAIN, out=gain)

        # Weighted sum of per-band gains gives one gain per FFT bin.  The band
        # weights are a partition of unity, so a unit gain everywhere leaves
        # the carrier untouched.
        combined_gain = gain @ weights
        output_spectrum = carrier_spectrum * combined_gain

        output_frames = np.fft.irfft(output_spectrum, n=frame, axis=1)
        output_frames *= self._synthesis_window[None, :]

        output = np.zeros(padded_length, dtype=np.float64)
        for offset in range(0, frame, hop):
            width = min(hop, frame - offset)
            chunk = output_frames[:, offset : offset + width]
            if width == hop:
                output[offset : offset + frame_count * hop] += chunk.reshape(-1)
            else:
                for row in range(frame_count):
                    start = row * hop + offset
                    output[start : start + width] += chunk[row]

        result = output[frame : frame + length]
        result = np.clip(result, -1.0, 1.0)
        return result.tolist()

    def cola_error(self):
        """Maximum deviation of the window overlap-add sum from one."""
        return float(self._cola_error)
