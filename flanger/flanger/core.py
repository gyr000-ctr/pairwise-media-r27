"""flanger 核心 API（初始未实现）。"""


class Flanger:
    def __init__(self, rate, min_ms, max_ms, fb):
        """构造"""
        raise NotImplementedError

    def process(self, samples, mix):
        """处理"""
        raise NotImplementedError

    def notch_spacing(self, delay_samples):
        """梳状间隔"""
        raise NotImplementedError
