# vocoder

## 题面（可直接粘贴）

子目录：vocoder。通道声码器。约束：分析与合成窗必须互补（COLA，窗和恒定 ≤1e-9）；带通滤波器组对数分布且相邻重叠；包络平滑用 attack/release 且跨帧连续；重建正弦测试信号 SNR ≥20 dB；100 万样本不超过 900 ms；非法带数可区分报错。验收：cd vocoder && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd vocoder && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：音视频处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
