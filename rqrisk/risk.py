# -*- coding: utf-8 -*-
# 版权所有 2021 深圳米筐科技有限公司（下称“米筐科技”）
#
# 除非遵守当前许可，否则不得使用本软件。
#
#     * 非商业用途（非商业用途指个人出于非商业目的使用本软件，或者高校、研究所等非营利机构出于教育、科研等目的使用本软件）：
#         遵守 Apache License 2.0（下称“Apache 2.0 许可”），
#         您可以在以下位置获得 Apache 2.0 许可的副本：http://www.apache.org/licenses/LICENSE-2.0。
#         除非法律有要求或以书面形式达成协议，否则本软件分发时需保持当前许可“原样”不变，且不得附加任何条件。
#
#     * 商业用途（商业用途指个人出于任何商业目的使用本软件，或者法人或其他组织出于任何目的使用本软件）：
#         未经米筐科技授权，任何个人不得出于任何商业目的使用本软件（包括但不限于向第三方提供、销售、出租、出借、转让本软件、
#         本软件的衍生产品、引用或借鉴了本软件功能或源代码的产品或服务），任何法人或其他组织不得出于任何目的使用本软件，
#         否则米筐科技有权追究相应的知识产权侵权责任。
#         在此前提下，对本软件的使用同样需要遵守 Apache 2.0 许可，Apache 2.0 许可与本许可冲突之处，以本许可为准。
#         详细的授权流程，请联系 public@ricequant.com 获取。

from __future__ import division

import warnings

import numpy as np
from statsmodels.formula.api import ols

from .utils import indicator_property, IndicatorProperty, annual_factor, safe_div, DAILY, deprecate_property


class Risk(object):
    def __init__(self, daily_returns, benchmark_daily_returns, risk_free_rate, period=DAILY):
        assert (len(daily_returns) == len(benchmark_daily_returns))
        self.period_count = len(daily_returns)

        self._portfolio = daily_returns
        self._benchmark = benchmark_daily_returns
        self._annual_factor = annual_factor(period)
        self._risk_free_rate = risk_free_rate
        self._risk_free_rate_per_period = (1 + risk_free_rate) ** (1 / self._annual_factor) - 1
        self._avg_excess_return = np.mean(daily_returns) - self._risk_free_rate_per_period
        self._active_returns = daily_returns - benchmark_daily_returns

    @indicator_property()
    def return_rate(self):
        return np.expm1(np.log1p(self._portfolio).sum())

    @indicator_property(min_period_count=1)
    def annual_return(self):
        return (1 + self.return_rate) ** (self._annual_factor / self.period_count) - 1

    @indicator_property()
    def benchmark_return(self):
        return np.expm1(np.log1p(self._benchmark).sum())

    @indicator_property(min_period_count=1)
    def benchmark_annual_return(self):
        return (1 + self.benchmark_return) ** (self._annual_factor / self.period_count) - 1

    @indicator_property()
    def arithmetic_excess_return(self):
        return self.return_rate - self.benchmark_return

    @indicator_property()
    def geometric_excess_return(self):
        return (1 + self.return_rate) / (1 + self.benchmark_return) - 1

    @indicator_property()
    def geometric_excess_annual_return(self):
        return (1 + self.annual_return) / (1 + self.benchmark_annual_return) - 1

    @indicator_property(min_period_count=2)
    def alpha(self):
        # annualized Jensen's alpha：https://en.wikipedia.org/wiki/Jensen%27s_alpha
        return self.annual_return - self._risk_free_rate - self.beta * (
            self.benchmark_annual_return - self._risk_free_rate
        )

    @indicator_property(min_period_count=2)
    def alpha_t_value(self):
        if all(self._benchmark):
            return np.nan
        result = ols("p ~ b", data={"p": self._portfolio, "b": self._benchmark}).fit()
        return result.tvalues["Intercept"]

    @indicator_property(min_period_count=2)
    def alpha_p_value(self):
        if all(self._benchmark):
            return np.nan
        result = ols("p ~ b", data={"p": self._portfolio, "b": self._benchmark}).fit()
        return result.pvalues["Intercept"]

    @indicator_property(min_period_count=2)
    def beta(self):
        cov = np.cov(np.vstack([self._portfolio, self._benchmark]), ddof=1)
        return safe_div(cov[0][1], cov[1][1])

    @indicator_property(min_period_count=2, value_when_pc_not_satisfied=0.)
    def volatility(self):
        return self._portfolio.std(ddof=1)

    @indicator_property()
    def annual_volatility(self):
        return self.volatility * (self._annual_factor ** 0.5)

    @indicator_property(min_period_count=2, value_when_pc_not_satisfied=0.)
    def benchmark_volatility(self):
        return self._benchmark.std(ddof=1)

    @indicator_property()
    def benchmark_annual_volatility(self):
        return self.benchmark_volatility * (self._annual_factor ** 0.5)

    @staticmethod
    def _calc_cum(returns):
        """ 计算累计净值 """
        returns = [0] + list(returns)
        df_cum = np.exp(np.log1p(returns).cumsum())
        return df_cum

    @classmethod
    def _calc_max_drawdown(cls, cum_nav):
        max_nav = np.maximum.accumulate(cum_nav)
        return abs(((cum_nav - max_nav) / max_nav).min())

    @indicator_property()
    def max_drawdown(self):
        cum_returns = self._calc_cum(self._portfolio)
        return self._calc_max_drawdown(cum_returns)

    @indicator_property(min_period_count=2, value_when_pc_not_satisfied=0.)
    def tracking_error(self):
        if np.all(np.isnan(self._benchmark)):
            return np.nan
        return self._active_returns.std(ddof=1)

    @indicator_property()
    def annual_tracking_error(self):
        if np.all(np.isnan(self._benchmark)):
            return np.nan
        return self.tracking_error * (self._annual_factor ** 0.5)

    @indicator_property(min_period_count=2)
    def information_ratio(self):
        # residual_return / residual_risk
        residual_returns = self._portfolio - self.beta * self._benchmark
        annual_residual_std = residual_returns.std(ddof=1) * np.sqrt(self._annual_factor)
        if not annual_residual_std:
            return np.nan
        return (self.annual_return - self.beta * self.benchmark_annual_return) / annual_residual_std

    @indicator_property(min_period_count=2)
    def sharpe(self):
        std_excess_return = np.sqrt((1 / (len(self._portfolio) - 1)) * np.sum(
            (self._portfolio - self._risk_free_rate_per_period - self._avg_excess_return) ** 2
        ))
        return safe_div(np.sqrt(self._annual_factor) * self._avg_excess_return, std_excess_return)

    @indicator_property()
    def excess_sharpe(self):
        # sharpe ratio of active returns
        return safe_div(np.sqrt(self._annual_factor) * np.mean(self._active_returns), self.tracking_error)

    @indicator_property(min_period_count=2, value_when_pc_not_satisfied=0.)
    def downside_risk(self):
        diff = self._portfolio - self._portfolio.mean()
        diff[diff > 0] = 0.
        return (np.sum(np.square(diff)) / (len(diff) - 1)) ** 0.5

    @indicator_property()
    def annual_downside_risk(self):
        return self.downside_risk * (self._annual_factor ** 0.5)

    @indicator_property()
    def sortino(self):
        return safe_div(self._annual_factor * self._avg_excess_return, self.annual_downside_risk)

    @indicator_property()
    def calmar(self):
        if np.isclose(self.max_drawdown, 0):
            return np.inf * np.sign(self.annual_return)
        else:
            return self.annual_return / self.max_drawdown

    @indicator_property(min_period_count=1)
    def _excess_return_rate(self):
        return np.expm1(np.log1p(self._active_returns).sum())

    @deprecate_property
    def excess_return_rate(self):
        # activate return rate
        return self._excess_return_rate

    @deprecate_property
    def excess_annual_return(self):
        return self._excess_annual_return

    @indicator_property(min_period_count=1)
    def _excess_annual_return(self):
        # active annual return
        return (1 + self._excess_return_rate) ** safe_div(self._annual_factor, self.period_count) - 1

    @indicator_property(min_period_count=2, value_when_pc_not_satisfied=0.)
    def excess_volatility(self):
        # volatility of active returns
        return self._active_returns.std(ddof=1)

    @indicator_property()
    def excess_annual_volatility(self):
        return self.excess_volatility * (self._annual_factor ** 0.5)

    @indicator_property(min_period_count=1)
    def geometric_excess_drawdown(self):
        excess = self._calc_cum(self._portfolio) / self._calc_cum(self._benchmark)
        return self._calc_max_drawdown(excess)

    @indicator_property(min_period_count=1)
    def excess_max_drawdown(self):
        cum_returns = self._calc_cum(self._active_returns)
        return self._calc_max_drawdown(cum_returns)

    @indicator_property()
    def var(self):
        """ default: 95% VaR """
        return self.param_var(0.05)

    def param_var(self, alpha):
        # 假设价格服从对数正态分布，则收益率亦应服从对数正态分布：https://stats.stackexchange.com/questions/378047/log-normal-returns
        import scipy.stats as stats
        log_return = np.log1p(self._portfolio)
        mean = np.mean(log_return)
        std = np.std(log_return)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "invalid value encountered in multiply")
            return np.expm1(-stats.norm(mean, std).ppf(alpha))

    @indicator_property(min_period_count=1)
    def win_rate(self):
        return len(self._portfolio[self._portfolio > 0]) / self.period_count

    @indicator_property(min_period_count=1)
    def excess_win_rate(self):
        return len(self._portfolio[self._portfolio > self._benchmark]) / self.period_count

    @indicator_property(min_period_count=2)
    def correlation(self):
        # 相关系数算法参考 https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html?highlight=corr#numpy.corrcoef
        return np.corrcoef(self._portfolio, self._benchmark)[0][1]

    @classmethod
    def _calc_ulcer_index(cls, returns):
        """ 累计回撤深度  相关计算公式参考：http://www.tangotools.com/ui/ui.htm """
        cum = cls._calc_cum(returns)
        drawdown_squared = ((cum / np.maximum.accumulate(cum) - 1) * 100) ** 2
        return (np.sum(drawdown_squared) / len(cum)) ** 0.5

    @indicator_property()
    def ulcer_index(self):
        return self._calc_ulcer_index(self._portfolio)

    @indicator_property()
    def excess_ulcer_index(self):
        return self._calc_ulcer_index(self._active_returns)

    def _calc_ulcer_performance_index(self, returns, ulcer_index):
        """ 累计回撤夏普率 相关计算公式参考：http://www.tangotools.com/ui/ui.htm """
        _return_rate = np.expm1(np.log1p(returns - self._risk_free_rate_per_period).sum())
        return safe_div(_return_rate, ulcer_index)

    @indicator_property()
    def ulcer_performance_index(self):
        return self._calc_ulcer_performance_index(self._portfolio, self.ulcer_index)

    @indicator_property()
    def excess_ulcer_performance_index(self):
        return self._calc_ulcer_performance_index(self._active_returns, self.excess_ulcer_index)

    def all(self):
        return {k: getattr(self, k) for k, v in self.__class__.__dict__.items() if isinstance(v, IndicatorProperty)}
