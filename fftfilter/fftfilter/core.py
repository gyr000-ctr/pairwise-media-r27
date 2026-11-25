"""fftfilter 核心 API（初始未实现）。"""


class FftFilter:
    def __init__(self, rate):
        """构造"""
        raise NotImplementedError

    def notch(self, samples, f0, q):
        """陷波"""
        raise NotImplementedError

    def response(self, f0, q, f):
        """幅频响应"""
        raise NotImplementedError
