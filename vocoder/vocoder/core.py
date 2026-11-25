"""vocoder 核心 API（初始未实现）。"""


class Vocoder:
    def __init__(self, rate, bands, frame):
        """构造"""
        raise NotImplementedError

    def process(self, carrier, modulator):
        """处理"""
        raise NotImplementedError

    def cola_error(self):
        """窗和误差"""
        raise NotImplementedError
