# -*- coding: utf-8 -*-
#
# Copyright 2016 Ricequant, Inc
#

from __future__ import division

import numpy as np
import scipy.stats as stats

APPROX_BDAYS_PER_MONTH = 21
APPROX_BDAYS_PER_YEAR = 252

MONTHS_PER_YEAR = 12
WEEKS_PER_YEAR = 52

DAILY = 'daily'
WEEKLY = 'weekly'
MONTHLY = 'monthly'
YEARLY = 'yearly'

ANNUALIZATION_FACTORS = {
    DAILY: APPROX_BDAYS_PER_YEAR,
    WEEKLY: WEEKS_PER_YEAR,
    MONTHLY: MONTHS_PER_YEAR
}


def _annual_factor(period):
    try:
        return ANNUALIZATION_FACTORS[period]
    except KeyError:
        raise ValueError("period cannot be {}, possible values: {}".format(
            period, ", ".join(ANNUALIZATION_FACTORS.keys())))

class Risk:
    def __init__(self, daily_returns, benchmark_daily_returns, risk_free_rate, period=DAILY):
        assert(len(daily_returns) == len(benchmark_daily_returns))

        self._portfolio = daily_returns
        self._benchmark = benchmark_daily_returns
        self._risk_free_rate = risk_free_rate
        self._annual_factor = _annual_factor(period)

        self._alpha = None
        self._beta = None
        self._sharpe = None
        self._return = np.expm1(np.log1p(self._portfolio).sum())
        self._annual_return = (1 + self._return) ** (self._annual_factor / len(self._portfolio)) - 1
        self._benchmark_return = np.expm1(np.log1p(self._benchmark).sum())
        self._benchmark_annual_return = (1 + self._benchmark_return) ** (self._annual_factor / len(self._portfolio)) - 1
        self._max_drawdown = None
        self._volatility = None
        self._annual_volatility = None
        self._information_ratio = None
        self._sortino = None
        self._tracking_error = None
        self._downside_risk = None
        self._calmar = None
        self._var = None

    @property
    def return_rate(self):
        return self._return

    @property
    def annual_return(self):
        return self._annual_return

    @property
    def benchmark_return(self):
        return self._benchmark_return

    @property
    def benchmark_annual_return(self):
        return self._benchmark_annual_return

    @property
    def alpha(self):
        if self._alpha is not None:
            return self._alpha

        if len(self._portfolio) < 2:
            self._alpha = np.nan
            self._beta = np.nan
            return np.nan

        self._alpha = self._annual_return - self._risk_free_rate - self.beta * (self._benchmark_annual_return - self._risk_free_rate)
        return self._alpha

    @property
    def beta(self):
        if self._beta is not None:
            return self._beta

        if len(self._portfolio) < 2:
            self._beta = np.nan
            return self._beta

        cov = np.cov(np.vstack([
            self._portfolio.values,
            self._benchmark.values
        ]), ddof=1)
        self._beta = cov[0][1] / cov[1][1]
        return self._beta

    def _calc_volatility(self):
        if len(self._portfolio) < 2:
            self._volatility = 0
            self._annual_volatility = 0
        else:
            std = self._portfolio.std()
            self._volatility = std * (len(self._portfolio) ** 0.5)
            self._annual_volatility = std * (self._annual_factor ** 0.5)

    @property
    def volatility(self):
        if self._volatility is not None:
            return self._volatility

        self._calc_volatility()
        return self._volatility

    @property
    def annual_volatility(self):
        if self._annual_volatility is not None:
            return self._annual_volatility

        self._calc_volatility()
        return self._annual_volatility

    @property
    def max_drawdown(self):
        if self._max_drawdown is not None:
            return self._max_drawdown

        if len(self._portfolio) < 1:
            self._max_drawdown = np.nan
            return np.nan

        df_cum = np.exp(np.log1p(self._portfolio).cumsum())
        max_return = df_cum.cummax()
        self._max_drawdown = df_cum.sub(max_return).div(max_return).min()
        return self._max_drawdown

    @property
    def tracking_error(self):
        if self._tracking_error is not None:
            return self._tracking_error

        if len(self._portfolio) < 2:
            self._tracking_error = np.nan
            return np.nan

        active_return = self._portfolio - self._benchmark
        self._tracking_error = np.std(active_return, ddof=1)
        return self._tracking_error

    @property
    def information_ratio(self):
        if self._information_ratio is not None:
            return self._information_ratio

        if len(self._portfolio) < 2:
            self._information_ratio = np.nan
            return np.nan

        if np.isnan(self.tracking_error):
            self._information_ratio = 0.0
            return 0

        if self.tracking_error == 0:
            self._information_ratio = np.nan
            return np.nan

        self._information_ratio = np.mean(self._portfolio - self._benchmark) / self.tracking_error
        return self._information_ratio

    @property
    def sharpe(self):
        if self._sharpe is not None:
            return self._sharpe

        self._sharpe = (self._annual_return - self._risk_free_rate) / self.volatility
        return self._sharpe

    @property
    def downside_risk(self):
        if self._downside_risk is not None:
            return self._downside_risk

        diff = self._portfolio - self._benchmark
        diff[diff > 0] = 0
        mean_squares = np.mean(np.square(diff))
        self._downside_risk = np.sqrt(mean_squares) * np.sqrt(self._annual_factor)
        return self._downside_risk

    @property
    def sortino(self):
        if self._sortino is not None:
            return self._sortino

        self._sortino = (self._annual_return - self._risk_free_rate) / self.downside_risk
        return self._sortino

    @property
    def calmor(self):
        if self._calmar is not None:
            return self._calmar

        max_dd = self.max_drawdown
        if max_dd < 0:
            tmp = self._annual_return / -max_dd
            if np.isinf(tmp):
                self._calmar = np.nan
            else:
                self._calmar = tmp
        else:
            self._calmar = np.nan

        return self._calmar

    @property
    def var(self):
        """ default: 95% VaR """
        if self._var is not None:
            return self._var

        self._var = self.param_var(0.05)
        return self._var

    def all(self):
        result = {
            'return': self.return_rate,
            'annual_return': self.annual_return,
            'benchmark_return': self.benchmark_return,
            'benchmark_annual_return': self.benchmark_annual_return,
            'alpha': self.alpha,
            'beta': self.beta,
            'sharpe': self.sharpe,
            'max_drawdown': self.max_drawdown,
            'volatility': self.volatility,
            'annual_volatility': self.annual_volatility,
            'information_ratio': self.information_ratio,
            'downside_risk': self.downside_risk,
            'sortino': self.sortina,
            'tracking_error': self.tracking_error,
            'calmar': self.calmor,
            'VaR': self.var,
        }

        #now all are done, _portfolio, _benchmark not needed now
        self._portfolio = None
        self._benchmark = None

    def param_var(self, alpha):
        log_return = np.log1p(self._portfolio)
        mean = np.mean(log_return)
        std = np.std(log_return)
        return -stats.norm(mean, std).ppf(alpha)


