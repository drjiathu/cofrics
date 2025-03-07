# Cofrics - Quantitative Investment Performance Analysis Toolkit

Cofrics is a Python toolkit for quantitative investment strategy performance analysis, providing a rich set of investment performance metrics.

## Key Features

### Basic Return Calculations

- Simple returns calculation (`simple_returns`)
- Cumulative returns calculation (`cum_returns`)
- Annualized return calculation (`annual_return`, `cagr`)

### Risk Metrics

- Maximum drawdown (`max_drawdown`)
- Annualized volatility (`annual_volatility`)
- Downside risk (`downside_risk`)
- Value at Risk (VaR) calculation (`value_at_risk`)
- Conditional Value at Risk (CVaR) calculation (`conditional_value_at_risk`)

### Performance Metrics

- Sharpe ratio (`sharpe_ratio`)
- Sortino ratio (`sortino_ratio`)
- Calmar ratio (`calmar_ratio`)
- Omega ratio (`omega_ratio`)
- Up capture ratio (`up_capture`)
- Down capture ratio (`down_capture`)

### Factor Analysis

- Alpha/Beta calculation (`alpha_beta`)
- Rolling Alpha/Beta calculation (`roll_alpha_beta`)
- Factor exposure calculation (`compute_exposures`)

### Other Tools

- Annualization factor calculation (`annualization_factor`)
- Returns aggregation (`aggregate_returns`)
- Time series stability analysis (`stability_of_timeseries`)

## Installation

Using pip:

```bash
pip install cofrics
```

Or using Poetry:

```bash
poetry add cofrics
```

## Usage Example

```python
import cofrics as cf
import pandas as pd

# Sample returns data
returns = pd.Series([0.01, -0.02, 0.03, 0.015, -0.01])

# Calculate annualized return
annual_ret = cf.annual_return(returns)

# Calculate Sharpe ratio
sharpe = cf.sharpe_ratio(returns)

print(f"Annualized Return: {annual_ret:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
