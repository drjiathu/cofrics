# Cofrics - 量化投资绩效分析工具包

Cofrics 是一个用于量化投资策略绩效分析的 Python 工具包，提供了丰富的投资绩效指标计算功能。

## 主要功能

### 基础收益计算

- 简单收益率计算 (`simple_returns`)
- 累计收益率计算 (`cum_returns`)
- 年化收益率计算 (`annual_return`, `cagr`)

### 风险指标

- 最大回撤 (`max_drawdown`)
- 年化波动率 (`annual_volatility`)
- 下行风险 (`downside_risk`)
- VaR (在险价值) 计算 (`value_at_risk`)
- CVaR (条件在险价值) 计算 (`conditional_value_at_risk`)

### 绩效指标

- 夏普比率 (`sharpe_ratio`)
- 索提诺比率 (`sortino_ratio`)
- 卡玛比率 (`calmar_ratio`)
- Omega 比率 (`omega_ratio`)
- 上行捕获率 (`up_capture`)
- 下行捕获率 (`down_capture`)

### 因子分析

- Alpha/Beta 计算 (`alpha_beta`)
- 滚动 Alpha/Beta 计算 (`roll_alpha_beta`)
- 因子暴露度计算 (`compute_exposures`)

### 其他工具

- 收益率年化因子计算 (`annualization_factor`)
- 收益率聚合 (`aggregate_returns`)
- 时间序列稳定性分析 (`stability_of_timeseries`)

## 安装

使用 pip 安装：

```bash
pip install cofrics
```

或者使用 Poetry：

```bash
poetry add cofrics
```

## 使用示例

```python
import cofrics as cf
import pandas as pd

# 示例收益率数据
returns = pd.Series([0.01, -0.02, 0.03, 0.015, -0.01])

# 计算年化收益率
annual_ret = cf.annual_return(returns)

# 计算夏普比率
sharpe = cf.sharpe_ratio(returns)

print(f"年化收益率: {annual_ret:.2%}")
print(f"夏普比率: {sharpe:.2f}")
```

## 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 本项目
2. 创建新的功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详情请参阅 [LICENSE](LICENSE) 文件。
