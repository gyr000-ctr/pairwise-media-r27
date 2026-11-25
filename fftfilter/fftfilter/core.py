"""fftfilter 核心 API：基于 FFT 的零相位陷波（带阻）滤波器。

设计要点
--------
* 频域窗 H 为实、偶对称窗：陷波中心增益为 0（衰减无穷大），过渡带采用
  Hann 斜坡，通带增益恒为 1（通带纹波 0 dB）。
* ``notch`` 采用零相位实现：正向 FFT→乘窗→IFFT 一次，再将信号反转后
  重复一次并反转回来（前后向各一次），不引入任何相位偏移。
* 朴素 ``dft``/``idft`` 与 ``fft``/``ifft`` 并列暴露，便于小规模交叉对照。
* 全部运算使用函数内局部数组，实例自身只读，多线程并发调用安全。
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "FftFilter",
    "FftFilterError",
    "InvalidRateError",
    "InvalidFrequencyError",
    "InvalidQError",
    "NonFiniteInputError",
    "dft",
    "idft",
    "fft",
    "ifft",
    "notch_window",
    "filter_kernel",
    "circular_convolve",
]


# ---------------------------------------------------------------------------
# 可区分的错误类型（均为 ValueError 子类，兼容宽泛的 ValueError 捕获）
# ---------------------------------------------------------------------------


class FftFilterError(ValueError):
    """fftfilter 参数错误基类。"""


class InvalidRateError(FftFilterError):
    """采样率非法（非正数或非有限值）。"""


class InvalidFrequencyError(FftFilterError):
    """频率非法（越界或非有限值）。"""


class InvalidQError(FftFilterError):
    """品质因数 Q 非法（非正数或非有限值）。"""


class NonFiniteInputError(FftFilterError):
    """输入样本中包含 NaN/Inf。"""


_REAL_TYPES = (int, float, np.integer, np.floating)


def _as_real(value, name, exc):
    """把入参校验为有限实数；类型问题抛 TypeError，值域问题抛 exc。"""
    if isinstance(value, (complex, np.complexfloating)):
        raise TypeError(f"{name} 必须为实数，不能为复数类型 {type(value).__name__}")
    if isinstance(value, bool) or not isinstance(value, _REAL_TYPES):
        raise TypeError(f"{name} 必须为实数，实际类型为 {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value):
        raise exc(f"{name} 必须为有限值，实际值为 {value}")
    return value


# ---------------------------------------------------------------------------
# 朴素 DFT / IDFT（仅用于小规模对照）与 FFT 封装
# ---------------------------------------------------------------------------


def _check_1d(arr, name):
    if arr.ndim != 1:
        raise ValueError(f"{name} 必须为一维序列，实际维度为 {arr.ndim}")
    return arr


def dft(x):
    """朴素 O(N²) 离散傅里叶变换，返回长度 N 的复数 numpy 数组。"""
    arr = _check_1d(np.asarray(x, dtype=np.float64), "x")
    n = arr.shape[0]
    if n > 8192:
        raise ValueError("朴素 DFT 仅支持 N<=8192 的小规模对照，请使用 fft()")
    idx = np.arange(n)
    basis = np.exp(-2j * math.pi * idx.reshape(n, 1) * idx / n)
    return basis @ arr


def idft(x):
    """朴素 O(N²) 离散傅里叶逆变换。"""
    arr = _check_1d(np.asarray(x, dtype=np.complex128), "x")
    n = arr.shape[0]
    if n > 8192:
        raise ValueError("朴素 IDFT 仅支持 N<=8192 的小规模对照，请使用 ifft()")
    idx = np.arange(n)
    basis = np.exp(2j * math.pi * idx.reshape(n, 1) * idx / n)
    return (basis @ arr) / n


def fft(x):
    """快速傅里叶变换（numpy pocketfft 封装）。"""
    return np.fft.fft(_check_1d(np.asarray(x, dtype=np.float64), "x"))


def ifft(x):
    """快速傅里叶逆变换（numpy pocketfft 封装）。"""
    return np.fft.ifft(_check_1d(np.asarray(x, dtype=np.complex128), "x"))


# ---------------------------------------------------------------------------
# 陷波频域窗
# ---------------------------------------------------------------------------


def _gain(dist, bandwidth):
    """距陷波中心 dist 处的单遍窗增益（标量或数组均可）。

    阻带半宽 = 0.1*bandwidth（增益 0），过渡带半宽 = 0.5*bandwidth，
    过渡带为 Hann 斜坡，之外增益恒为 1（通带零纹波）。
    """
    dist = np.asarray(dist, dtype=np.float64)
    stop_w = 0.1 * bandwidth
    pass_w = 0.5 * bandwidth
    gain = np.where(dist <= stop_w, 0.0, 1.0)
    band = (dist > stop_w) & (dist < pass_w)
    t = (dist - stop_w) / (pass_w - stop_w)  # 过渡带内归一化坐标，取值 (0, 1)
    return np.where(band, 0.5 * (1.0 - np.cos(math.pi * t)), gain)


def notch_window(rate, n, f0, q):
    """构造长度 n 的实偶对称陷波频域窗 H。

    参数
    ----
    rate : 采样率（Hz），必须为正有限实数。
    n    : FFT 点数（非负整数）。
    f0   : 陷波中心频率（Hz），必须满足 0 < f0 < rate/2。
    q    : 品质因数，必须为正有限实数；带宽 bandwidth = f0 / q。
    """
    rate = _as_real(rate, "rate", InvalidRateError)
    if rate <= 0.0:
        raise InvalidRateError(f"rate 必须为正数，实际值为 {rate}")
    f0 = _as_real(f0, "f0", InvalidFrequencyError)
    if not 0.0 < f0 < rate / 2.0:
        raise InvalidFrequencyError(
            f"f0 必须落在 (0, rate/2={rate / 2.0}) 内，实际值为 {f0}"
        )
    q = _as_real(q, "q", InvalidQError)
    if q <= 0.0:
        raise InvalidQError(f"q 必须为正数，实际值为 {q}")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0:
        raise TypeError(f"n 必须为非负整数，实际类型为 {type(n).__name__}")

    n = int(n)
    window = np.ones(n, dtype=np.float64)
    if n == 0:
        return window

    k = np.arange(n)
    # 折叠到 [0, rate/2]，自动保证 H 偶对称（实信号零相位窗）
    folded = np.minimum(k, n - k) * (rate / n)
    bandwidth = f0 / q
    window = _gain(np.abs(folded - f0), bandwidth)
    window[0] = 1.0  # 直流 bin 始终保留，保护信号均值
    return window


def filter_kernel(rate, n, f0, q):
    """陷波窗对应的实值冲激响应 h = real(IFFT(H))。"""
    return np.real(np.fft.ifft(notch_window(rate, n, f0, q)))


def circular_convolve(x, h):
    """等长一维序列的圆周卷积（频域实现，取实部）。"""
    xa = _check_1d(np.asarray(x, dtype=np.float64), "x")
    ha = _check_1d(np.asarray(h, dtype=np.float64), "h")
    if xa.shape != ha.shape:
        raise ValueError("circular_convolve 要求两个序列等长")
    return np.real(np.fft.ifft(np.fft.fft(xa) * np.fft.fft(ha)))


# ---------------------------------------------------------------------------
# 主 API
# ---------------------------------------------------------------------------


class FftFilter:
    """零相位 FFT 陷波（带阻）滤波器。

    实例只保存不可变的采样率，:meth:`notch` 不持有任何可变共享状态，
    可被多个线程并发调用。
    """

    def __init__(self, rate):
        rate = _as_real(rate, "rate", InvalidRateError)
        if rate <= 0.0:
            raise InvalidRateError(f"rate 必须为正数，实际值为 {rate}")
        self.rate = rate

    def _validate_notch_params(self, f0, q):
        f0 = _as_real(f0, "f0", InvalidFrequencyError)
        if not 0.0 < f0 < self.rate / 2.0:
            raise InvalidFrequencyError(
                f"f0 必须落在 (0, rate/2={self.rate / 2.0}) 内，实际值为 {f0}"
            )
        q = _as_real(q, "q", InvalidQError)
        if q <= 0.0:
            raise InvalidQError(f"q 必须为正数，实际值为 {q}")
        return f0, q

    @staticmethod
    def _collect_samples(samples):
        if isinstance(samples, (str, bytes, bytearray)):
            raise TypeError("samples 必须为数值序列，不能为字符串/字节串")
        try:
            seq = list(samples)
        except TypeError as exc:
            raise TypeError("samples 必须为可迭代的一维数值序列") from exc
        return seq

    def notch(self, samples, f0, q):
        """对 samples 执行零相位陷波滤波，返回等长 Python float 列表。

        流程：参数/有限性校验 → 构造频域窗 → 正向 FFT 乘窗 IFFT，
        反转后再做一遍并反转回来（前后向各一次，零相位）。
        不修改输入序列。
        """
        f0, q = self._validate_notch_params(f0, q)
        seq = self._collect_samples(samples)
        if len(seq) == 0:
            return []

        try:
            arr = np.array(seq, dtype=np.float64, copy=True)
        except (TypeError, ValueError) as exc:
            raise TypeError("samples 中包含无法转换为实数的元素") from exc
        if arr.ndim != 1:
            raise ValueError("samples 必须为一维数值序列")
        if not np.all(np.isfinite(arr)):
            raise NonFiniteInputError("samples 中包含 NaN 或 Inf，拒绝滤波")

        n = arr.shape[0]
        window = notch_window(self.rate, n, f0, q)

        def one_pass(x):
            spectrum = np.fft.fft(x)
            return np.real(np.fft.ifft(spectrum * window))

        # 前后向各一次：第二遍在反转信号上进行，整体相位抵消
        y = one_pass(arr)
        y = one_pass(y[::-1])[::-1]

        # 返回精确的两遍频域滤波结果（与 IFFT/卷积严格一致，不做任何截断）。
        # 陷波核 L1 范数天然大于 1，单位幅值的病态输入可能产生 >1 的输出，
        # 这是真实滤波器的振铃特性而非缺陷；截断反而会破坏题面 1e-6 一致性。
        return y.tolist()

    def response(self, f0, q, f):
        """返回零相位滤波器在频率 f 处的幅频响应（实浮点数）。

        零相位实现等价于单遍窗的两遍级联，故幅度响应为单遍增益的平方。
        合法求值区间为 0 <= f <= rate/2。
        """
        f0, q = self._validate_notch_params(f0, q)
        f = _as_real(f, "f", InvalidFrequencyError)
        if not 0.0 <= f <= self.rate / 2.0:
            raise InvalidFrequencyError(
                f"f 必须落在 [0, rate/2={self.rate / 2.0}] 内，实际值为 {f}"
            )
        if f == 0.0:  # 直流 bin 受保护，始终单位增益
            return 1.0
        bandwidth = f0 / q
        gain = float(_gain(abs(f - f0), bandwidth))
        return gain * gain
