# flanger

## 题面（可直接粘贴）

子目录：flanger。镶边效果（梳状滤波）。约束：延迟在最小值以上的调制范围内（不得 <1 采样）；反馈系数 |fb|<1 保证稳定（否则报错）；梳状陷波频率间隔 = rate/delay（≤1e-6）；干湿比为等功率混合；100 万样本不超过 400 ms；并发安全。验收：cd flanger && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd flanger && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：音视频处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
