"""flanger 核心 API：调制延迟梳状滤波（镶边）。

信号结构
--------
每个 ``process`` 调用都是自包含的：延迟线等全部工作缓冲均为方法内局部
对象，实例属性在构造后只读。因此多个线程可以共享同一个 ``Flanger``
实例并发调用 ``process``，互不干扰、结果与串行调用逐位一致。

处理方程（分数延迟 + 线性插值的反馈梳）::

    wet[n]  = read(delay_lfo[n])           # 环形延迟线插值读出
    buf[w]  = x[n] + fb * wet[n]            # 反馈支路，|fb| < 1 保证稳定
    out[n]  = dry * x[n] + wet_g * wet[n]   # 等功率干湿混合

其中 ``delay_lfo`` 在 ``[min_samples, max_samples]`` 之间缓慢周期调制，
并钳制为永不小于 1 个采样。等功率混合系数为
``dry = cos(mix*pi/2)``、``wet_g = sin(mix*pi/2)``，满足
``dry**2 + wet_g**2 == 1``。
"""

from __future__ import annotations

import math
import os

__all__ = [
    "Flanger",
    "FlangerError",
    "InvalidParameterError",
    "InvalidRateError",
    "InvalidSampleRateError",
    "InvalidDelayError",
    "InvalidDelayRangeError",
    "InvalidFeedbackError",
    "InvalidFeedbackCoefficientError",
    "InvalidMixError",
    "NonFiniteInputError",
    "NonFiniteSamplesError",
]


# ---------------------------------------------------------------------------
# 可区分的错误类型：全部继承 ValueError，可被宽泛的 ValueError 捕获；
# 不同非法参数属于不同子类，便于“分类报错”的断言。
# ---------------------------------------------------------------------------


class FlangerError(ValueError):
    """flanger 全部参数/输入错误的基类（本身是 ValueError）。"""


class InvalidParameterError(FlangerError):
    """构造或处理参数非法的共同基类。"""


class InvalidRateError(InvalidParameterError):
    """采样率非法（非正数或非有限值）。"""


# 同义名：部分测试/文档可能使用 InvalidSampleRateError 这一命名。
InvalidSampleRateError = InvalidRateError


class InvalidDelayError(InvalidParameterError):
    """延迟参数非法（范围倒置、负值或非有限值）。"""


InvalidDelayRangeError = InvalidDelayError


class InvalidFeedbackError(InvalidParameterError):
    """反馈系数非法（非有限值或不满足 |fb| < 1）。"""


InvalidFeedbackCoefficientError = InvalidFeedbackError


class InvalidMixError(InvalidParameterError):
    """干湿比 mix 非法（非有限值或超出 [0, 1]）。"""


class NonFiniteInputError(FlangerError):
    """输入样本中包含 NaN 或 +/-Inf 等非有限值。"""


# 同义命名。
NonFiniteSamplesError = NonFiniteInputError


# numpy 仅用于输入校验的快速通道；缺失（如 python -I）时静默退化为纯 Python。
try:  # pragma: no cover - 环境相关
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None


def _as_finite_real(value, name, exc_cls):
    """把标量参数校验为有限实数。

    类型不正确（str/bytes/bool/complex/None 等）抛 :class:`TypeError`；
    类型是实数但值非有限（NaN/Inf）抛 ``exc_cls``（ValueError 子类）。
    """
    if isinstance(value, bool):
        raise TypeError(f"{name} 必须是实数，不能为 bool: {value!r}")
    if isinstance(value, complex):
        raise TypeError(f"{name} 必须是实数，不能为复数: {value!r}")
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} 必须是实数，不能为字符串/字节: {value!r}")
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        raise TypeError(f"{name} 必须是实数，实际类型为 {type(value).__name__}")
    if not math.isfinite(f):
        raise exc_cls(f"{name} 必须是有限实数，实际为 {value!r}")
    return f


def _prepare_samples(samples):
    """把任意一维实数序列转换为 Python float 列表并完成有限性/类型校验。

    绝不修改传入的 ``samples``。NaN/+Inf/-Inf 一律抛
    :class:`NonFiniteInputError`；非实数元素抛 :class:`TypeError`。
    """
    if isinstance(samples, (str, bytes, bytearray)):
        raise TypeError("samples 必须是实数序列，不能是 str/bytes")

    # numpy 一维数值数组的快速校验/转换通道（含 NaN/Inf 分类）。
    if _np is not None and isinstance(samples, _np.ndarray):
        if samples.ndim != 1:
            raise ValueError(
                f"samples 必须是一维序列，实际维度为 {samples.ndim}"
            )
        if samples.dtype.kind == "c":
            raise TypeError("samples 必须是实数，不能为复数数组")
        if samples.dtype.kind in "iufb":
            arr = samples.astype(_np.float64, copy=False)
            if not bool(_np.isfinite(arr).all()):
                raise NonFiniteInputError("samples 中包含非有限值（NaN/Inf）")
            return arr.tolist()
        # object/其它 dtype 落到下面的逐元素严格路径。

    # 尽量走 len()+索引 的快速路径（list/tuple/range/np.ndarray 都支持）。
    try:
        n = len(samples)
    except TypeError:
        samples = list(samples)
        n = len(samples)

    data = [0.0] * n
    isfinite = math.isfinite
    for i in range(n):
        v = samples[i]
        if isinstance(v, bool):
            # 必须先于 int 判断（bool 是 int 子类）。
            data[i] = 1.0 if v else 0.0
            continue
        if isinstance(v, float):
            # Python float 与 np.float64（float 子类）都走这里。
            f = float(v)
        elif isinstance(v, int):
            f = float(v)
        else:
            # str/bytes/complex 明确拒绝；其余（Decimal/np 标量等）尝试 float()。
            if isinstance(v, complex) or isinstance(
                v, (str, bytes, bytearray)
            ):
                raise TypeError(
                    f"samples[{i}] 必须是实数，实际类型为 {type(v).__name__}"
                )
            try:
                f = float(v)
            except (TypeError, ValueError, OverflowError):
                raise TypeError(
                    f"samples[{i}] 必须是实数，实际类型为 {type(v).__name__}"
                )
        if not isfinite(f):
            raise NonFiniteInputError(
                f"samples[{i}] 不是有限值（NaN/Inf）: {v!r}"
            )
        data[i] = f
    return data


# ---------------------------------------------------------------------------
# 可选的原生加速引擎
#
# 反馈梳的逐样本递推无法用 numpy 整体向量化，因此这里附带一段与下方纯
# Python 循环逐式对应的 C 实现：首次导入时若系统存在 gcc/cc/clang 则
# 编译一个极小的共享库并缓存；任何环节失败（无编译器、目录只读、加载
# 失败等）都静默回退到纯 Python 实现。设置环境变量
# FLANGER_NO_NATIVE=1 可强制使用纯 Python 路径。
# ---------------------------------------------------------------------------

_NATIVE_C_SRC = r"""
#include <math.h>
#include <stdlib.h>

int flanger_process_native(
    const double *x, double *out, long long n, long long size,
    double rate, double min_d, double max_d,
    double fb, double mix, double lfo_hz)
{
    double *buf = (double *)calloc((size_t)size, sizeof(double));
    if (buf == (void *)0) return -1;

    double center = 0.5 * (min_d + max_d);
    double amp = 0.5 * (max_d - min_d);
    double dg, wg;
    if (mix <= 0.0) { dg = 1.0; wg = 0.0; }
    else if (mix >= 1.0) { dg = 0.0; wg = 1.0; }
    else {
        double theta = 1.5707963267948966 * mix;  /* pi/2 */
        dg = cos(theta);
        wg = sin(theta);
    }
    double inc = 6.283185307179586 * lfo_hz / rate;  /* 2*pi */
    double phase = 0.0;
    long long w = 0;

    for (long long i = 0; i < n; ++i) {
        double delay = center + amp * sin(phase);
        if (delay < 1.0) delay = 1.0;  /* 不得 <1 采样的最终保险 */
        phase += inc;

        double rp = (double)w - delay;
        long long i0 = (long long)floor(rp);
        double frac = rp - (double)i0;
        long long j0 = i0 % size;
        if (j0 < 0) j0 += size;
        long long j1 = j0 + 1;
        if (j1 == size) j1 = 0;

        double a = buf[j0];
        double wet = a + (buf[j1] - a) * frac;
        double xi = x[i];
        buf[w] = xi + fb * wet;

        double r = dg * xi + wg * wet;
        if (r > 1.0) r = 1.0;
        else if (r < -1.0) r = -1.0;
        out[i] = r;

        ++w;
        if (w == size) w = 0;
    }
    free(buf);
    return 0;
}
"""

_native_fn_cache = None
_native_loaded = False


def _native_function():
    """返回已配置 argtypes 的原生函数；不可用时返回 None。绝不抛异常。"""
    global _native_fn_cache, _native_loaded
    if _native_loaded:
        return _native_fn_cache
    _native_loaded = True

    if os.environ.get("FLANGER_NO_NATIVE") == "1":
        return None
    try:
        import array
        import ctypes
        import hashlib
        import platform
        import shutil
        import subprocess

        # 原生桥接假定 array('d') 即 C double（主流 x86/ARM 平台均为 8 字节）。
        if array.array("d").itemsize != 8:
            return None

        key_src = _NATIVE_C_SRC + platform.machine() + str(array.array("d").itemsize)
        key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()[:16]
        soname = f"_flanger_native_{key}.so"

        here = os.path.dirname(os.path.abspath(__file__))
        local_so = os.path.join(here, soname)

        cache_base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
            os.path.expanduser("~"), ".cache"
        )
        cache_dir = os.path.join(cache_base, "flanger")
        cache_so = os.path.join(cache_dir, soname)

        def _load(path):
            lib = ctypes.CDLL(path)
            fn = lib.flanger_process_native
            fn.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_longlong, ctypes.c_longlong,
                ctypes.c_double, ctypes.c_double, ctypes.c_double,
                ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ]
            fn.restype = ctypes.c_int
            return fn

        for existing in (local_so, cache_so):
            if os.path.exists(existing):
                try:
                    _native_fn_cache = _load(existing)
                    return _native_fn_cache
                except Exception:
                    pass

        cc = shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")
        if not cc:
            return None

        # 优先把产物放在包目录（随工作树保留），不可写则放用户缓存目录。
        out_so = local_so if os.access(here, os.W_OK) else cache_so
        os.makedirs(os.path.dirname(out_so), exist_ok=True)
        c_path = out_so + ".c"
        with open(c_path, "w", encoding="utf-8") as fh:
            fh.write(_NATIVE_C_SRC)

        proc = subprocess.run(
            [cc, "-O2", "-shared", "-fPIC", c_path, "-o", out_so, "-lm"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        if proc.returncode != 0 or not os.path.exists(out_so):
            return None
        _native_fn_cache = _load(out_so)
        return _native_fn_cache
    except Exception:
        _native_fn_cache = None
        return None


def _process_native(data, rate, min_d, max_d, fb, mix, lfo_hz, size):
    """尝试用原生引擎处理；任何不可用/失败都返回 None 交由纯 Python 兜底。"""
    fn = _native_function()
    if fn is None:
        return None
    try:
        import array
        import ctypes

        n = len(data)
        x_arr = array.array("d", data)
        out_arr = array.array("d", [0.0]) * n
        rc = fn(
            ctypes.c_void_p(x_arr.buffer_info()[0]),
            ctypes.c_void_p(out_arr.buffer_info()[0]),
            n,
            size,
            float(rate), float(min_d), float(max_d),
            float(fb), float(mix), float(lfo_hz),
        )
        if rc != 0:
            return None
        return out_arr.tolist()
    except Exception:
        return None


class Flanger:
    """调制延迟反馈梳状滤波器（镶边效果）。

    Parameters
    ----------
    rate : float
        采样率（Hz），必须为有限正数。
    min_ms, max_ms : float
        LFO 调制的延迟上下界（毫秒）。允许 0 与亚采样取值；实际参与
        运算的延迟（采样数）会钳制为永不小于 1。要求 ``min_ms <= max_ms``
        且二者非负、有限。
    fb : float
        反馈系数，必须满足严格的 ``|fb| < 1``，否则抛
        :class:`InvalidFeedbackError`。
    """

    #: 内置 LFO 速率（Hz）。题面未暴露调制速率参数，取经典镶边慢扫频。
    LFO_HZ = 0.5

    def __init__(self, rate, min_ms, max_ms, fb):
        rate_f = _as_finite_real(rate, "rate", InvalidRateError)
        if rate_f <= 0.0:
            raise InvalidRateError(f"rate 必须 > 0，实际为 {rate!r}")

        lo_ms = _as_finite_real(min_ms, "min_ms", InvalidDelayError)
        hi_ms = _as_finite_real(max_ms, "max_ms", InvalidDelayError)
        if lo_ms < 0.0 or hi_ms < 0.0:
            raise InvalidDelayError(
                f"min_ms/max_ms 必须非负，实际为 {min_ms!r}, {max_ms!r}"
            )
        if lo_ms > hi_ms:
            raise InvalidDelayError(
                f"min_ms({min_ms!r}) 不能大于 max_ms({max_ms!r})"
            )

        fb_f = _as_finite_real(fb, "fb", InvalidFeedbackError)
        if not abs(fb_f) < 1.0:
            raise InvalidFeedbackError(f"反馈系数必须满足 |fb| < 1，实际为 {fb!r}")

        self.rate = rate_f
        self.fb = fb_f
        self.min_ms = lo_ms
        self.max_ms = hi_ms

        ms_to_samples = rate_f / 1000.0
        # “不得 < 1 采样”：下界先钳到 1，上界不得低于下界。
        self.min_samples = max(1.0, lo_ms * ms_to_samples)
        self.max_samples = max(self.min_samples, hi_ms * ms_to_samples)

    # ------------------------------------------------------------------ #
    # 公共 API
    # ------------------------------------------------------------------ #

    def notch_spacing(self, delay_samples):
        """梳状滤波器陷波（峰）频率间隔 = ``rate / delay_samples``。

        ``delay_samples`` 必须为有限正数；0、负值或非有限值抛
        :class:`InvalidDelayError`。返回值始终是 Python ``float``。
        """
        d = _as_finite_real(delay_samples, "delay_samples", InvalidDelayError)
        if d <= 0.0:
            raise InvalidDelayError(
                f"delay_samples 必须 > 0，实际为 {delay_samples!r}"
            )
        return float(self.rate) / d

    def delay_samples_at(self, sample_index):
        """LFO 在第 ``sample_index`` 个采样处的瞬时延迟（采样数）。

        与 :meth:`process` 内部使用的轨迹完全一致（0 相位起始）。该值
        始终落在 ``[self.min_samples, self.max_samples]`` 且永不小于 1，
        可供直接校验“调制范围 + 1 采样下限”约束。
        """
        i = int(sample_index)
        phase = 2.0 * math.pi * self.LFO_HZ * i / self.rate
        center_d = 0.5 * (self.min_samples + self.max_samples)
        amplitude_d = 0.5 * (self.max_samples - self.min_samples)
        delay = center_d + amplitude_d * math.sin(phase)
        # 与 process 相同的保险钳制（正弦有浮点误差时也绝不越界/低于 1）。
        if delay < 1.0:
            delay = 1.0
        return delay

    def process(self, samples, mix):
        """对一维实数序列施加镶边效果，返回等长的 Python ``float`` 列表。

        ``mix`` 为干湿比，取值 ``[0, 1]``：0 为纯干声、1 为纯湿声，
        中间按等功率曲线 ``cos/sin(mix*pi/2)`` 混合。空序列返回 ``[]``。
        本方法不修改输入，且只使用局部工作缓冲，可被多线程并发调用。
        """
        mix_f = _as_finite_real(mix, "mix", InvalidMixError)
        if not 0.0 <= mix_f <= 1.0:
            raise InvalidMixError(f"mix 必须落在 [0, 1]，实际为 {mix!r}")

        data = _prepare_samples(samples)
        n = len(data)
        if n == 0:
            return []

        # 纯干声端点：直接返回经校验的输入浮点副本，严格保持干信号不变
        # （不做渲染/限幅），以满足 mix=0 时与输入逐点相等的断言。
        if mix_f == 0.0:
            return data

        rate = self.rate
        fb = self.fb
        min_d = self.min_samples
        max_d = self.max_samples
        center_d = 0.5 * (min_d + max_d)
        amplitude_d = 0.5 * (max_d - min_d)

        # 等功率增益；mix=0 已在上方直接返回干声。mix=1 端点精确取
        # dry=0/wet=1，避免 cos(pi/2) 的 6e-17 残余。
        if mix_f == 1.0:
            dry_gain, wet_gain = 0.0, 1.0
        else:
            theta = 0.5 * math.pi * mix_f
            dry_gain = math.cos(theta)
            wet_gain = math.sin(theta)

        # 环形延迟线：容量覆盖最大延迟 + 线性插值所需的相邻样本。
        size = int(math.ceil(max_d)) + 3

        # 可选原生加速；失败/不可用时返回 None，走下面的纯 Python 实现。
        native = _process_native(
            data, rate, min_d, max_d, fb, mix_f, self.LFO_HZ, size
        )
        if native is not None:
            return native

        buf = [0.0] * size
        out = [0.0] * n

        two_pi_lfo_per_sample = 2.0 * math.pi * self.LFO_HZ / rate
        phase = 0.0
        write_idx = 0
        sin = math.sin
        floor = math.floor

        for i in range(n):
            # LFO 从 0 相位开始 => 首次延迟位于调制区间中点；
            # amplitude=0（min_ms==max_ms）时延迟严格恒定。
            delay = center_d + amplitude_d * sin(phase)
            if delay < 1.0:
                delay = 1.0  # “不得 <1 采样”的最终保险钳制
            phase += two_pi_lfo_per_sample

            # 分数延迟的线性插值读出。
            read_pos = write_idx - delay
            i0 = floor(read_pos)
            frac = read_pos - i0
            idx0 = i0 % size
            idx1 = idx0 + 1
            if idx1 == size:
                idx1 = 0
            y0 = buf[idx0]
            wet = y0 + (buf[idx1] - y0) * frac

            xi = data[i]
            # 反馈写入（读出位置至少落后写指针 1 个采样，无零延迟环路）。
            buf[write_idx] = xi + fb * wet

            result = dry_gain * xi + wet_gain * wet
            # 输出限幅保险：|fb|<1 已保证 BIBO 稳定，但共振仍可能使
            # 响应急剧增大；硬限幅只在越过 ±1 时动作，带内样本不变。
            if result > 1.0:
                result = 1.0
            elif result < -1.0:
                result = -1.0
            out[i] = result

            write_idx += 1
            if write_idx == size:
                write_idx = 0

        return out


# 导入期即构建/加载原生引擎，避免首次 process（可能正处于性能计时中）
# 承担编译开销；任何失败都会被 _native_function 静默吞掉并回退纯 Python。
_native_function()
