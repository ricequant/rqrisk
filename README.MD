# RQRISK
simple and fast indicator calculator


## Install

pip install rqrisk [--upgrade]


## Usage

import packages

`
import rqrisk
`
* create Risk object

`
risk = rqrisk.Risk(portfolio_returns, benchmark_returns, risk_free_rate, period)
`

* get result

`
result = risk.all() # dict
`

## Example

```jupyter
In [1]: import rqrisk

In [2]: import numpy

In [3]: risk = rqrisk.Risk(
   ...:     # 组合日收益率
   ...:     daily_returns=numpy.array([0.0006, 0.0213, -0.0155, -0.017, -0.0025, -0.0173, -0.0006, -0.0094, 0.0108, -0.0151]),
   ...:     # 基准日收益率
   ...:     benchmark_daily_returns=numpy.array([0.0024, 0.0048, -0.0036, -0.013, 0.0107, -0.0203, 0.0097, 0.0018, 0.0083, -0.0089]),
   ...:     # 无风险利率
   ...:     risk_free_rate=0.02
   ...: )
   
In [4]: risk.all()
Out[4]: 
{'return_rate': -0.04454119145160703,
 'annual_return': -0.6827938423763955,
 'benchmark_return': -0.008557443372075783,
 'benchmark_annual_return': -0.19472831925009826,
 'alpha': -0.9467029842994686,
 'beta': 0.8912056741947679,
 'volatility': 0.013005985801417234,
 'annual_volatility': 0.2064636239147226,
 'benchmark_volatility': 0.010414781162687322,
 'benchmark_annual_volatility': 0.1653295254937847,
 'max_drawdown': 0.06502900516774743,
 'tracking_error': 0.009180922248518028,
 'annual_tracking_error': 0.14574262245479186,
 'information_ratio': -0.4113963917368945,
 'sharpe': -5.55274570048971,
 'excess_sharpe': -6.328416385440682,
 'downside_risk': 0.011371747671783752,
 'annual_downside_risk': 0.1805208978703054,
 'sortino': -6.350732870959105,
 'calmar': -10.499835275275627,
 'excess_return_rate': -0.036369224165807304,
 'excess_annual_return': -0.6068586934419611,
 'excess_volatility': 0.009180922248518028,
 'excess_annual_volatility': 0.14574262245479186,
 'excess_max_drawdown': 0.050301584825935344,
 'var': 0.025161857442751474,
 'win_rate': 0.3,
 'correlation': 0.7136492542281812}
```
