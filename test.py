#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy.testing import assert_almost_equal
import pandas as pd
import numpy as np

import rqrisk
from rqrisk import DAILY, WEEKLY, MONTHLY, NATURAL_DAILY


# Simple benchmark, no drawdown
simple_benchmark = pd.Series(
    np.array([1., 1., 1., 1., 1., 1., 1., 1., 1.]) / 100,
    index=pd.date_range('2000-1-30', periods=9, freq='D'))

zero_benchmark = pd.Series(
    np.array([0]*9),
    index=pd.date_range('2000-1-30', periods=9, freq='D'))

simple_weekly_benchamrk = pd.Series(
    np.array([1., 1., 1., 1., 1., 1., 1., 1., 1.]) / 100,
    index=pd.date_range('2000-1-30', periods=9, freq='W'))

simple_monthly_benchamrk = pd.Series(
    np.array([1., 1., 1., 1., 1., 1., 1., 1., 1.]) / 100,
    index=pd.date_range('2000-1-30', periods=9, freq='M'))

# All positive returns, small variance
positive_returns = pd.Series(
    np.array([1., 2., 1., 1., 1., 1., 1., 1., 1.]) / 100,
    index=pd.date_range('2000-1-30', periods=9, freq='D'))

# All negative returns
negative_returns = pd.Series(
    np.array([0., -6., -7., -1., -9., -2., -6., -8., -5.]) / 100,
    index=pd.date_range('2000-1-30', periods=9, freq='D'))

# Weekly returns
weekly_returns = pd.Series(
    np.array([0., 1., 10., -4., 2., 3., 2., 1., -10.])/100,
    index=pd.date_range('2000-1-30', periods=9, freq='W'))

# Monthly returns
monthly_returns = pd.Series(
    np.array([0., 1., 10., -4., 2., 3., 2., 1., -10.])/100,
    index=pd.date_range('2000-1-30', periods=9, freq='M'))

# Series of length 1
one_return = pd.Series(
    np.array([1.])/100,
    index=pd.date_range('2000-1-30', periods=1, freq='D'))

one_benchmark = pd.Series(
    np.array([1.])/100,
    index=pd.date_range('2000-1-30', periods=1, freq='D'))

dot_one_benchmark = np.array([1.])/10


volatile_returns = pd.Series(
    np.array([-3, 1, 4, 5, -10, -1, 2, 0.5, 1]) / 100,
    index=pd.date_range('2000-1-30', periods=9, freq='D')
)

volatile_benchmark = pd.Series(
    np.array([1, 2, -5, 3, 10, -3, -1, 4, 1]) / 100,
    index=pd.date_range('2000-1-30', periods=9, freq='D')
)

volatile_weekly_benchmark = pd.Series(
    np.array([1, 2, -5, 3, 10, -3, -1, 4, 1]) / 100,
    index=pd.date_range('2000-1-30', periods=9, freq='W')
)

volatile_monthly_benchmark = pd.Series(
    np.array([1, 2, -5, 3, 10, -3, -1, 4, 1]) / 100,
    index=pd.date_range('2000-1-30', periods=9, freq='M')
)


def _r(returns, benchmark_returns, risk_free_rate, period=DAILY):
    if benchmark_returns is None:
        benchmark_returns = pd.Series([np.nan] * len(returns), index=returns.index, dtype=returns.dtype)
    return rqrisk.Risk(returns, benchmark_returns, risk_free_rate, period)


def test_return():
    assert_almost_equal(
        rqrisk.Risk(positive_returns, simple_benchmark, 0).max_drawdown,
        -0.0)
    assert_almost_equal(
        rqrisk.Risk(negative_returns, simple_benchmark, 0).max_drawdown,
        0.36590730349873601)
    assert_almost_equal(
        rqrisk.Risk(one_return, one_benchmark, 0).max_drawdown,
        0)


def test_annual_return():
    def _assert(returns, period, desired_annual_return):
        assert_almost_equal(_r(returns, None, 0, period).annual_return, desired_annual_return)

    _assert(weekly_returns, WEEKLY, 0.24690830513998208)
    _assert(monthly_returns, MONTHLY, 0.052242061386048144)
    _assert(pd.Series([], dtype=float), DAILY, np.nan)


def test_beta_alpha():
    def _assert(returns, benchmark, risk_free_rate, period, desired_beta, desired_alpha):
        r = _r(returns, benchmark, risk_free_rate, period)
        assert_almost_equal(r.beta, desired_beta)
        assert_almost_equal(r.alpha, desired_alpha)

    _assert(one_return, one_benchmark, 0, DAILY, np.nan, np.nan)
    _assert(positive_returns, volatile_benchmark, 0.0252, DAILY, 0.004444444444444445, 15.050183341344455)
    _assert(volatile_returns, volatile_benchmark, 0.0252, DAILY, -0.6755555555555558, 14.515193843282699)
    _assert(volatile_returns, volatile_benchmark, 0.052, WEEKLY, -0.6755555555555558, 0.4543018988394533)
    _assert(volatile_returns, volatile_benchmark, 0.024, MONTHLY, -0.6755555555555558, 0.05117674068461497)


def test_calmar():
    assert_almost_equal(
        rqrisk.Risk(one_return, one_benchmark, 0).calmar,
        np.inf)
    assert_almost_equal(
        rqrisk.Risk(weekly_returns, simple_weekly_benchamrk, 0, rqrisk.WEEKLY).calmar,
        2.4690830513998208)
    assert_almost_equal(
        rqrisk.Risk(monthly_returns, simple_monthly_benchamrk, 0, rqrisk.MONTHLY).calmar,
        0.52242061386048144)


def test_annual_volatity():
    assert_almost_equal(
        rqrisk.Risk(simple_benchmark, simple_benchmark, 0).annual_volatility,
        0)
    assert_almost_equal(
        rqrisk.Risk(weekly_returns, simple_weekly_benchamrk, 0, rqrisk.WEEKLY).annual_volatility,
        0.38851569394870583)
    assert_almost_equal(
        rqrisk.Risk(monthly_returns, simple_monthly_benchamrk, 0, rqrisk.MONTHLY).annual_volatility,
        0.18663690238892558)


def test_volatility():
    def _assert(returns, period, desired_v, desired_annual_v):
        r = _r(returns, None, 0, period)
        assert_almost_equal(r.volatility, desired_v)
        assert_almost_equal(r.annual_volatility, desired_annual_v),

    _assert(one_return, DAILY, 0, 0)
    _assert(positive_returns, DAILY, 0.0033333333333333335, 0.052915026221291815)
    _assert(negative_returns, DAILY, 0.03179797338056485, 0.5047771785649584)
    _assert(volatile_returns, DAILY, 0.04433145359423463, 0.7037400088100719)


def test_excess_volatility():
    def _assert(returns, benchmark, period, desired_excess_v, desired_excess_annual_v):
        r = _r(returns, benchmark, 0, period)
        assert_almost_equal(r.excess_volatility, desired_excess_v)
        assert_almost_equal(r.excess_annual_volatility, desired_excess_annual_v)

    _assert(one_return, one_benchmark, DAILY, 0, 0)
    _assert(positive_returns, zero_benchmark, DAILY, 0.0033333333333333335, 0.052915026221291815)
    _assert(negative_returns, zero_benchmark, DAILY, 0.03179797338056485, 0.5047771785649584)
    _assert(volatile_returns, volatile_benchmark, DAILY, 0.07983489907998326, 1.2673397334574499)


def test_sharpe():
    def _assert(returns, risk_free_rate, period, desired_sharpe):
        assert_almost_equal(_r(returns, None, risk_free_rate, period).sharpe, desired_sharpe)

    _assert(one_return, 0, DAILY,  np.nan)
    _assert(positive_returns, 0, DAILY, 52.915026221291804)
    _assert(negative_returns, 0, DAILY, -24.406808633910085)
    _assert(volatile_returns, 0.0252, DAILY, -0.2343037804431006)
    _assert(weekly_returns, 0.052, WEEKLY, 0.613028149736571)
    _assert(monthly_returns, 0.036, MONTHLY, 0.16742323233212023)


def test_downside_risk():
    def _assert(returns, risk_free_rate, period, desired_downside_risk, desired_annual_downside_risk):
        r = _r(returns, None, risk_free_rate, period)
        assert_almost_equal(r.downside_risk, desired_downside_risk)
        assert_almost_equal(r.annual_downside_risk, desired_annual_downside_risk)

    _assert(one_return, 0, DAILY, 0., 0.)
    _assert(weekly_returns, 0, WEEKLY, 0.04069420743199846, 0.2934501030208764)
    _assert(weekly_returns, 0.052, WEEKLY, 0.04069420743199846, 0.2934501030208764)
    _assert(monthly_returns, 0, MONTHLY, 0.04069420743199846, 0.14096886969193667)
    _assert(monthly_returns, 0.036, MONTHLY, 0.04069420743199846, 0.14096886969193667)


def test_sortino():
    def _assert(returns, risk_free_rate, period, desired_sortino):
        assert_almost_equal(_r(returns, None, risk_free_rate, period).sortino, desired_sortino)

    _assert(one_return, 0, DAILY, np.nan)
    _assert(positive_returns, 0, DAILY, 158.74507866387563)
    _assert(negative_returns, 0.0252, DAILY, -37.99176785990703)
    _assert(weekly_returns, 0, WEEKLY, 0.9844565938637175)
    _assert(weekly_returns, 0.052, WEEKLY, 0.8116236953171263)
    _assert(monthly_returns, 0, MONTHLY, 0.47291765062992475)
    _assert(monthly_returns, 0.036, MONTHLY, 0.22166137487442497)


def test_tracking_error_information_ratio():
    def _assert(returns, benchmark, period, desired_te, desired_annual_te, desired_excess_sharpe):
        r = _r(returns, pd.Series(benchmark.values, index=returns.index), 0, period)
        assert_almost_equal(r.tracking_error, desired_te)
        assert_almost_equal(r.annual_tracking_error, desired_annual_te)
        assert_almost_equal(r.excess_sharpe, desired_excess_sharpe)

    _assert(positive_returns, zero_benchmark, DAILY, 0.0033333333333333335, 0.052915026221291815, 52.915026221291804)
    _assert(positive_returns, simple_benchmark, DAILY, 0.003333333333333333, 0.05291502622129181, 5.291502622129182)
    _assert(negative_returns, simple_benchmark, DAILY, 0.03179797338056485, 0.5047771785649584, -29.399110399937154)
    _assert(weekly_returns, simple_benchmark, WEEKLY, 0.05387743291748205, 0.38851569394870583, -0.5948565648975399)


def test_information_ratio():
    def _assert(returns, benchmark, period, desired_ir):
        r = _r(returns, pd.Series(benchmark.values, index=returns.index), 0, period)
        assert_almost_equal(r.information_ratio, desired_ir)
    _assert(positive_returns, zero_benchmark, DAILY, np.nan)
    _assert(positive_returns, volatile_benchmark, DAILY, 285.37186618887614)
    _assert(volatile_returns, volatile_benchmark, DAILY, 27.529981978194222)
    _assert(weekly_returns, volatile_weekly_benchmark, WEEKLY, 1.7335510771535123)


def test_max_drawdown():
    def _assert(returns, benchmark, desired_max_dd, desired_excess_max_dd):
        r = _r(returns, benchmark, 0, DAILY)
        assert_almost_equal(r.max_drawdown, desired_max_dd)
        assert_almost_equal(r.excess_max_drawdown, desired_excess_max_dd)

    _assert(volatile_returns, zero_benchmark, 0.10899999999999994, 0.10899999999999994)
    _assert(volatile_returns, volatile_benchmark, 0.10899999999999994, 0.20000000000000007)


def test_natural_daily():
    """ 测试自然日 """
    r = _r(volatile_returns, volatile_benchmark, 0, NATURAL_DAILY)
    assert_almost_equal(r.annual_return, -0.41377394067925255)
    assert_almost_equal(r.benchmark_annual_return, 92.86171734511149)
    assert_almost_equal(r.alpha, 62.31947511024051)
    assert_almost_equal(r.information_ratio, 97.92626812366308)
    assert_almost_equal(r.sharpe, -0.23942084288518412)


def test_win_rate():
    """ 测试胜率 """
    def _assert(returns, benchmark, desired_win_rate):
        assert_almost_equal(_r(returns, benchmark, 0).win_rate, desired_win_rate)

    _assert(volatile_returns, volatile_benchmark, 0.6666666666666666)
    _assert(positive_returns, volatile_benchmark, 1)
    _assert(negative_returns, volatile_benchmark, 0)


def test_excess_win_rate():
    """ 测试超额胜率 """
    def _assert(returns, benchmark, desired_win_rate):
        assert_almost_equal(_r(returns, benchmark, 0).excess_win_rate, desired_win_rate)

    _assert(volatile_returns, volatile_benchmark, 0.4444444444444444)
    _assert(positive_returns, zero_benchmark, 1)
    _assert(negative_returns, zero_benchmark, 0)


def test_correlation():
    """ 测试相关系数 """
    def _assert(returns, benchmark, period):
        r = _r(returns, benchmark, 0, period)
        assert_almost_equal(r.correlation, returns.corr(benchmark))

    _assert(one_return, one_benchmark, DAILY)                       # np.nan
    _assert(positive_returns, simple_benchmark, DAILY)              # np.nan
    _assert(positive_returns, volatile_benchmark, DAILY)            # 0.05773502691896258
    _assert(negative_returns, simple_benchmark, DAILY)              # np.nan
    _assert(negative_returns, volatile_benchmark, DAILY)            # -0.32984900530449723
    _assert(weekly_returns, simple_weekly_benchamrk, WEEKLY)        # np.nan
    _assert(weekly_returns, volatile_weekly_benchmark, WEEKLY)      # -0.3411258215912419
    _assert(monthly_returns, simple_monthly_benchamrk, MONTHLY)     # np.nan
    _assert(monthly_returns, volatile_monthly_benchmark, MONTHLY)   # -0.3411258215912419
