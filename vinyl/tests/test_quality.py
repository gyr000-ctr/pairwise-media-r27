"""补充质量用例：分数延迟精度、噪声分布、并发与错误分类。"""

import math
import threading

import pytest

import vinyl

RATE = 48000
FS = float(RATE)


def _sine(freq, count, amplitude=0.5):
    return [amplitude * math.sin(2.0 * math.pi * freq * i / FS) for i in range(count)]


def _max_error(actual, expected, skip):
    return max(abs(a - b) for a, b in zip(actual[skip:-skip], expected[skip:-skip]))


@pytest.mark.parametrize("freq", [50, 440, 1000, 5000, 15000])
@pytest.mark.parametrize("depth_ms", [0.5, 2.0, 10.0])
def test_fractional_delay_matches_ideal(freq, depth_ms):
    """抖晃输出与解析的「理想带限延迟」逐点相差不超过 1e-3。"""
    count = 12000
    processor = vinyl.Vinyl(RATE, 0.5, depth_ms, 0.0, 1)
    samples = _sine(freq, count)
    delays = processor.delay_samples([float(i) for i in range(count)])
    reference = [
        0.5 * math.sin(2.0 * math.pi * freq * (i - d) / FS)
        for i, d in enumerate(delays)
    ]
    skip = int(depth_ms * FS / 1000.0) + 100
    assert _max_error(processor.process(samples), reference, skip) <= 1e-3


def test_half_sample_delay_kernel_symmetry():
    """半个采样点的延迟核左右对称、直流增益为 1。"""
    samples = [0.0] * 200
    samples[100] = 1.0
    kernel = vinyl.fractional_delay(samples, 0.5)
    span = [v for v in kernel if abs(v) > 1e-12]
    assert math.isclose(sum(span), 1.0, abs_tol=1e-12)
    # 峰值落在两个相邻整数采样点上且相等（半采样点延迟的对称性）。
    peak = max(range(len(kernel)), key=lambda i: kernel[i])
    assert abs(kernel[peak] - kernel[peak + 1]) < 1e-12


def test_constant_fractional_delay_preserves_signal():
    """常数分数延迟对直流信号只做时移，幅度不变。"""
    samples = [0.3] * 400
    shifted = vinyl.fractional_delay(samples, 3.7)
    assert all(abs(v - 0.3) < 1e-12 for v in shifted[60:-60])


def test_seed_reproducible_and_distribution():
    """底噪由种子逐位复现，并服从 N(0, noise^2)。"""
    processor = vinyl.Vinyl(RATE, 0.5, 0.0, 0.25, 12345)
    first = processor.process([0.0] * 100000)
    second = processor.process([0.0] * 100000)
    assert first == second

    mean = sum(first) / len(first)
    variance = sum((v - mean) ** 2 for v in first) / len(first)
    assert abs(mean) < 0.01
    assert math.isclose(math.sqrt(variance), 0.25, rel_tol=0.03)

    other = vinyl.Vinyl(RATE, 0.5, 0.0, 0.25, 12346).process([0.0] * 100000)
    assert other != first


def test_zero_depth_is_transparent():
    """深度为 0 时输出与输入逐位一致（含无噪声）。"""
    samples = [0.1, -0.2, 0.3, 0.0] * 50
    output = vinyl.Vinyl(RATE, 3.0, 0.0, 0.0, 1).process(samples)
    assert output == samples


def test_wow_rate_is_configurable():
    """延迟调制的主频等于 wow_hz，幅度等于 depth_ms。"""
    for wow_hz in (0.5, 2.0, 5.0):
        processor = vinyl.Vinyl(RATE, wow_hz, 2.0, 0.0, 0)
        count = int(FS * 8)
        delays = processor.delay_samples([float(i) for i in range(count)])
        assert math.isclose(max(delays) - min(delays), 2.0 * FS / 1000.0, abs_tol=1e-6)
        # 过零点计数：cos 调制在 8 秒内应有 2*wow_hz*8 个完整周期
        upward = sum(
            1
            for i in range(1, count)
            if delays[i - 1] < 1.0 and delays[i] >= 1.0
        )
        assert upward == pytest.approx(int(round(wow_hz * 8)), abs=1)


def test_errors_are_distinguishable():
    cases = [
        (lambda: vinyl.Vinyl(0, 0.5, 1.0, 0.0, 1), vinyl.InvalidRateError),
        (lambda: vinyl.Vinyl(RATE, 0.0, 1.0, 0.0, 1), vinyl.InvalidWowRateError),
        (lambda: vinyl.Vinyl(RATE, 0.5, -1.0, 0.0, 1), vinyl.InvalidDepthError),
        (lambda: vinyl.Vinyl(RATE, 0.5, 1.0, -0.1, 1), vinyl.InvalidNoiseError),
        (lambda: vinyl.Vinyl(RATE, 0.5, 1.0, 0.1, -1), vinyl.InvalidSeedError),
        (
            lambda: vinyl.Vinyl(RATE, 0.5, 1.0, 0.0, 1).process([float("nan")]),
            vinyl.NonFiniteInputError,
        ),
        (
            lambda: vinyl.Vinyl(RATE, 0.5, 1.0, 0.0, 1).process([float("inf")]),
            vinyl.NonFiniteInputError,
        ),
    ]
    for call, expected in cases:
        with pytest.raises(expected):
            call()
        assert issubclass(expected, ValueError)


def test_concurrent_calls_are_consistent():
    processor = vinyl.Vinyl(RATE, 0.5, 1.0, 0.05, 7)
    samples = _sine(440, 20000)
    expected = processor.process(samples)
    results = [None] * 6

    def worker(index):
        results[index] = processor.process(list(samples))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert all(result == expected for result in results)


def test_input_not_mutated_and_output_typed():
    samples = [0.1, 0.2, 0.3] * 40
    snapshot = list(samples)
    output = vinyl.Vinyl(RATE, 0.5, 2.0, 0.01, 3).process(samples)
    assert samples == snapshot
    assert isinstance(output, list)
    assert all(isinstance(v, float) and math.isfinite(v) for v in output)
    assert vinyl.Vinyl(RATE, 0.5, 2.0, 0.0, 3).process([]) == []
