## RQRISK
simple and fast indicator calculator

#
#### 1. Install

pip install rqrisk [--upgrade]

#
#### 2. Usage

*  import packages

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
