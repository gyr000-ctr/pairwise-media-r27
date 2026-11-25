# fftfilter

## 题面（可直接粘贴）

子目录：fftfilter。频域滤波（陷波/带阻）。约束：FFT 与朴素 DFT 对照（≤1e-9）；频域乘窗后 IFFT 实部重建与直接卷积一致（≤1e-6）；陷波中心频点幅度衰减 ≥30 dB 而通带纹波 ≤0.5 dB；零相位实现（前后向各一次）不引入相位偏移；100 万样本不超过 800 ms；非法频率可区分报错。验收：cd fftfilter && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd fftfilter && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：音视频处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
