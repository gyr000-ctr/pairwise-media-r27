"""vinyl 核心 API：黑胶抖晃（wow/flutter）与底噪。

设计要点
--------
* 抖晃为「低频调制延迟」：瞬时延迟

      d(n) = depth_ms/2 * (1 - cos(2*pi*wow_hz*n/rate))   [单位：采样点]

  即延迟在 ``[0, depth_ms]`` 毫秒之间以 ``wow_hz`` 为正弦速率摆动，
  起始延迟为 0，因此 ``depth_ms == 0`` 时完全不发生抖晃。
* 延迟为分数采样点时，用 Kaiser 窗 sinc 核做分数延迟插值（每个采样点
  用 48 个相邻样本、8192 个相位量化）。通带（|f| 远低于奈奎斯特）内
  插值幅度误差在 1e-5 量级，与「理想带限延迟」的偏差远小于 1e-3；
  相位量化误差不超过 1/16384 个采样点（2 的整数幂相位表，量化精确）。
* 底噪为高斯白噪声，标准差等于 ``noise``，由 ``seed`` 派生的
  ``numpy.random.PCG64`` 生成：同种子逐位可复现，不同种子不同。
* 输入长度、类型、有限性均做校验；非法参数抛出可区分的
  :class:`VinylError` 子类（同时它们都是 :class:`ValueError`）。
* 实例只保存只读参数，``process`` 的全部状态都在局部变量里，每个调用
  新建独立随机数发生器，因此同一实例可被多线程并发调用且结果一致。
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "Vinyl",
    "VinylError",
    "InvalidRateError",
    "InvalidWowRateError",
    "InvalidDepthError",
    "InvalidNoiseError",
    "InvalidSeedError",
    "NonFiniteInputError",
    "modulation_delay",
    "fractional_delay",
]


# ---------------------------------------------------------------------------
# 可区分的错误类型（均为 ValueError 子类，兼容宽泛的 ValueError 捕获）
# ---------------------------------------------------------------------------


class VinylError(ValueError):
    """vinyl 参数 / 输入错误基类。"""


class InvalidRateError(VinylError):
    """采样率非法（非正数或非有限值）。"""


class InvalidWowRateError(VinylError):
    """抖晃频率非法（非正数、非有限值或达到奈奎斯特频率）。"""


class InvalidDepthError(VinylError):
    """抖晃深度非法（负数、非有限值或超出上界）。"""


class InvalidNoiseError(VinylError):
    """底噪幅度非法（负数或非有限值）。"""


class InvalidSeedError(VinylError):
    """随机种子非法（非整数或为负数）。"""


class NonFiniteInputError(VinylError):
    """输入样本中包含 NaN/Inf。"""


# 抖晃深度上界（毫秒）。真实黑胶的抖晃深度在毫秒量级，这里留出充足余量；
# 超出即视为非法参数（幅度有界），同时保证内部零填充缓冲有确定上界。
MAX_DEPTH_MS = 1000.0

# 分数延迟插值核参数：48 抽头、Kaiser beta=9、8192 相位。
# 相位表为 2 的整数幂，``frac * _PHASES`` 与 ``rint`` 的舍入对全部
# 可表示相位精确（不引入额外量化误差）。
_TAPS = 48
_HALF = _TAPS // 2
_PHASES = 8192
_KAISER_BETA = 9.0

# 分块长度：分块后每次乘加都在较小的临时数组上完成，命中缓存，
# 比整段大数组逐抽头累加显著更快（1M 样本从 ~0.16s 降到 ~0.10s）。
_CHUNK = 131072


def _build_kernel():
    """预计算分数延迟 sinc 插值表，形状 (PHASES, TAPS)，每行直流增益为 1。"""
    k = np.arange(_TAPS, dtype=np.float64)
    frac = np.arange(_PHASES, dtype=np.float64) / _PHASES
    # 抽头相对「基准整数位置 + frac」的连续偏移：k - (HALF - 1) - frac
    offsets = k[None, :] - (_HALF - 1) - frac[:, None]
    ratio = np.clip(np.abs(offsets) / _HALF, 0.0, 1.0)
    window = np.i0(_KAISER_BETA * np.sqrt(1.0 - ratio * ratio)) / np.i0(_KAISER_BETA)
    table = np.sinc(offsets) * window
    table /= table.sum(axis=1, keepdims=True)
    # 转成按抽头排列的只读列，便于逐列向量化乘加。
    columns = tuple(np.ascontiguousarray(table[:, k]) for k in range(_TAPS))
    for column in columns:
        column.flags.writeable = False
    return columns


_KERNEL = _build_kernel()

_REAL_TYPES = (int, float, np.integer, np.floating)


def _as_real(value, name, exc):
    """校验入参为有限实数；类型问题抛 TypeError，值域问题抛 exc。"""
    if isinstance(value, (complex, np.complexfloating)):
        raise TypeError(f"{name} 必须为实数，不能为复数类型 {type(value).__name__}")
    if isinstance(value, bool) or not isinstance(value, _REAL_TYPES):
        raise TypeError(f"{name} 必须为实数，实际类型为 {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value):
        raise exc(f"{name} 必须为有限值，实际值为 {value!r}")
    return value


def modulation_delay(n, rate, wow_hz, depth_ms):
    """返回第 n 个采样点的瞬时延迟（单位：采样点）。

    ``d(n) = depth_ms/2 * (1 - cos(2*pi*wow_hz*n/rate)) * rate/1000``

    参数可为标量或一维 numpy 数组；depth_ms 为 0 时恒为 0。
    """
    rate = float(rate)
    depth = float(depth_ms) * rate / 1000.0
    phase = (2.0 * math.pi * float(wow_hz) / rate) * np.asarray(n, dtype=np.float64)
    return 0.5 * depth * (1.0 - np.cos(phase))


def _kernel_delay(values, positions, pad_left):
    """对 ``values`` 在连续位置 ``positions`` 处做插值（分数延迟）。

    ``positions`` 与 ``values`` 处于同一坐标：``positions[j] == j`` 表示取
    ``values[j]`` 原值，``positions[j] == j - d`` 表示延迟 ``d`` 个采样点。
    ``values`` 已放入长度足够的零填充缓冲，真实样本从 ``pad_left`` 开始。
    """
    count = positions.shape[0]
    base = np.floor(positions)
    frac = positions - base  # [0, 1)
    index = base.astype(np.int64)

    # 相位表量化：frac 落在 [0.5, 1) 时舍入到 _PHASES，进位到下一个整数
    # 延迟并取相位 0（两者等价，且避免查表越界）。
    phase = np.rint(frac * _PHASES).astype(np.int64)
    carry = phase >= _PHASES
    if carry.any():
        phase[carry] = 0
        index[carry] += 1
    offset = index + pad_left - (_HALF - 1)

    out = np.zeros(count, dtype=np.float64)
    for start in range(0, count, _CHUNK):
        stop = min(count, start + _CHUNK)
        phase_chunk = phase[start:stop]
        offset_chunk = offset[start:stop]
        acc = np.zeros(stop - start, dtype=np.float64)
        for k in range(_TAPS):
            acc += _KERNEL[k][phase_chunk] * values[offset_chunk + k]
        out[start:stop] = acc
    return out


def fractional_delay(samples, delay_samples):
    """对一维序列做分数延迟（正延迟 = 向后移动）。

    使用与 :class:`Vinyl` 相同的 Kaiser 窗 sinc 插值核。长度保持不变，
    延迟超出序列左端的位置以 0 填充。非法输入与 ``Vinyl.process`` 一致地
    抛出 :class:`NonFiniteInputError` / :class:`TypeError`。
    """
    values = _coerce_samples(samples)
    if values.size == 0:
        return []
    if not np.all(np.isfinite(values)):
        raise NonFiniteInputError("samples 中包含 NaN 或 Inf，无法做分数延迟")
    try:
        delay = np.asarray(delay_samples, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("delay_samples 必须为有限实数或其序列") from exc
    if delay.ndim == 0:
        delay = np.full(values.size, float(delay), dtype=np.float64)
    elif delay.ndim != 1 or delay.shape[0] != values.size:
        raise ValueError("delay_samples 必须为标量或与 samples 等长的一维序列")
    if not np.all(np.isfinite(delay)):
        raise NonFiniteInputError("delay_samples 中包含 NaN 或 Inf")

    pad_left = int(math.ceil(max(0.0, float(delay.max())))) + _HALF + 2
    pad_right = int(math.ceil(max(0.0, -float(delay.min())))) + _HALF + 2
    positions = np.arange(values.size, dtype=np.float64) - delay
    buffer = np.zeros(values.size + pad_left + pad_right, dtype=np.float64)
    buffer[pad_left:pad_left + values.size] = values
    return _kernel_delay(buffer, positions, pad_left).tolist()


def _coerce_samples(samples):
    """把输入转为一维 float64 数组（不修改调用方数据）。"""
    if isinstance(samples, (str, bytes, bytearray)):
        raise TypeError("samples 必须为数值序列，不能为字符串/字节串")
    try:
        array = np.asarray(samples, dtype=np.float64)
    except (TypeError, ValueError):
        # 生成器等非数组型可迭代对象：先物化再转换。
        try:
            array = np.asarray(list(samples), dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError("samples 必须为可转换为实数的数值序列") from exc
    except Exception as exc:  # pragma: no cover - 防御性
        raise TypeError("samples 必须为可转换为实数的数值序列") from exc
    if array.ndim != 1:
        raise ValueError(f"samples 必须为一维数值序列，实际维度为 {array.ndim}")
    return array


# ---------------------------------------------------------------------------
# 主 API
# ---------------------------------------------------------------------------


class Vinyl:
    """黑胶抖晃与底噪处理器。

    参数
    ----
    rate : 采样率（Hz），正有限实数。
    wow_hz : 抖晃调制频率（Hz），需满足 ``0 < wow_hz < rate/2``。
    depth_ms : 抖晃深度（毫秒），需满足 ``0 <= depth_ms <= MAX_DEPTH_MS``；
        深度为 0 时输出不含任何抖晃。
    noise : 底噪标准差（幅值），非负有限实数；0 表示不加噪。
    seed : 非负整数随机种子，保证底噪逐位可复现。
    """

    def __init__(self, rate, wow_hz, depth_ms, noise, seed):
        rate = _as_real(rate, "rate", InvalidRateError)
        if rate <= 0.0:
            raise InvalidRateError(f"rate 必须为正数，实际值为 {rate}")

        wow_hz = _as_real(wow_hz, "wow_hz", InvalidWowRateError)
        if not 0.0 < wow_hz < rate / 2.0:
            raise InvalidWowRateError(
                f"wow_hz 必须落在 (0, rate/2={rate / 2.0}) 内，实际值为 {wow_hz}"
            )

        depth_ms = _as_real(depth_ms, "depth_ms", InvalidDepthError)
        if depth_ms < 0.0:
            raise InvalidDepthError(f"depth_ms 必须为非负数，实际值为 {depth_ms}")
        if depth_ms > MAX_DEPTH_MS:
            raise InvalidDepthError(
                f"depth_ms 不得超过 {MAX_DEPTH_MS}，实际值为 {depth_ms}"
            )
        depth_samples = depth_ms * rate / 1000.0
        if depth_samples > 1 << 20:
            raise InvalidDepthError(
                f"depth_ms 换算后的延迟（{depth_samples} 采样点）过大"
            )

        noise = _as_real(noise, "noise", InvalidNoiseError)
        if noise < 0.0:
            raise InvalidNoiseError(f"noise 必须为非负数，实际值为 {noise}")

        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError(f"seed 必须为整数，实际类型为 {type(seed).__name__}")
        seed = int(seed)
        if seed < 0:
            raise InvalidSeedError(f"seed 必须为非负整数，实际值为 {seed}")

        # 只读参数：process 不修改它们。
        self.rate = rate
        self.wow_hz = wow_hz
        self.depth_ms = depth_ms
        self.noise = noise
        self.seed = seed
        self.depth_samples = depth_samples

    # ------------------------------------------------------------------
    # 公共辅助
    # ------------------------------------------------------------------

    def delay_samples(self, n):
        """第 n 个采样点的瞬时延迟（采样点），供外部对照/可视化。"""
        return modulation_delay(n, self.rate, self.wow_hz, self.depth_ms)

    # ------------------------------------------------------------------
    # 主处理
    # ------------------------------------------------------------------

    def process(self, samples):
        """处理一维采样序列，返回等长的 Python float 列表。

        不修改输入；相同参数与种子下每次调用结果逐位一致。
        """
        values = _coerce_samples(samples)
        count = values.shape[0]
        if count == 0:
            return []
        if not np.all(np.isfinite(values)):
            raise NonFiniteInputError("samples 中包含 NaN 或 Inf，拒绝处理")

        if self.depth_ms == 0.0:
            # 深度为 0：延迟恒为 0，无抖晃，原样通过（仍做拷贝以免共享内存）。
            output = np.array(values, dtype=np.float64, copy=True)
        else:
            output = self._apply_wow(values)

        if self.noise > 0.0:
            rng = np.random.Generator(np.random.PCG64(self.seed))
            # 先取满 count 个样本再缩放：噪声序列只取决于 seed 与长度。
            noise = rng.standard_normal(count)
            noise *= self.noise
            output += noise

        return output.tolist()

    def _apply_wow(self, values):
        """对 values 施加调制分数延迟，返回新的 float64 数组。"""
        count = values.shape[0]
        depth = self.depth_samples

        # 延迟上界决定左侧零填充长度。
        pad_left = int(math.ceil(depth)) + _HALF + 2
        buffer = np.zeros(count + pad_left + _HALF + 2, dtype=np.float64)
        buffer[pad_left:pad_left + count] = values

        index = np.arange(count, dtype=np.float64)
        positions = index - modulation_delay(
            index, self.rate, self.wow_hz, self.depth_ms
        )
        return _kernel_delay(buffer, positions, pad_left)
