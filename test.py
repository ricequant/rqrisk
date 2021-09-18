#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy.testing import assert_almost_equal
import pandas as pd
import numpy as np

import rqrisk
from rqrisk import DAILY, WEEKLY, MONTHLY


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


def _r(returns, benchmark_returns, risk_free_rate, period=DAILY):
    if benchmark_returns is None:
        benchmark_returns = pd.Series([np.nan] * len(returns), index=returns.index)
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
    assert_almost_equal(
        rqrisk.Risk(weekly_returns, simple_weekly_benchamrk, 0, rqrisk.WEEKLY).annual_return,
        0.24690830513998208)
    assert_almost_equal(
        rqrisk.Risk(monthly_returns, simple_monthly_benchamrk, 0, rqrisk.MONTHLY).annual_return,
        0.052242061386048144)


def test_beta_alpha():
    def _assert(returns, benchmark, risk_free_rate, period, desired_beta, desired_alpha):
        r = _r(returns, benchmark, risk_free_rate, period)
        assert_almost_equal(r.beta, desired_beta)
        assert_almost_equal(r.alpha, desired_alpha)

    _assert(positive_returns, volatile_benchmark, 0.0252, DAILY, 0.004444444444444445, 2.7599786666666666)
    _assert(volatile_returns, volatile_benchmark, 0.0252, DAILY, -0.6755555555555558, 2.087642666666667)
    _assert(volatile_returns, volatile_benchmark, 0.052, WEEKLY, -0.6755555555555558, 0.35236740740740746)
    _assert(volatile_returns, volatile_benchmark, 0.024, MONTHLY, -0.6755555555555558, 0.06120888888888891)


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
    _assert(volatile_returns, 0.0252, DAILY, -0.2347457838574942)
    _assert(weekly_returns, 0.052, WEEKLY, 0.6097279790199784)
    _assert(monthly_returns, 0.036, MONTHLY, 0.16431191406489135)


def test_downside_risk():
    def _assert(returns, risk_free_rate, period, desired_downside_risk, desired_annual_downside_risk):
        r = _r(returns, None, risk_free_rate, period)
        assert_almost_equal(r.downside_risk, desired_downside_risk)
        assert_almost_equal(r.annual_downside_risk, desired_annual_downside_risk)

    _assert(one_return, 0, DAILY, 0., 0.)
    _assert(weekly_returns, 0, WEEKLY, 0.03807886552931954, 0.2745906043549196)
    _assert(weekly_returns, 0.052, WEEKLY, 0.0385405630472623, 0.2779199525043137)
    _assert(monthly_returns, 0, MONTHLY, 0.03807886552931954, 0.1319090595827292)
    _assert(monthly_returns, 0.036, MONTHLY, 0.03947625868797599, 0.13674977148061343)


def test_sortino():
    def _assert(returns, risk_free_rate, period, desired_sortino):
        assert_almost_equal(_r(returns, None, risk_free_rate, period).sortino, desired_sortino)

    _assert(one_return, 0, DAILY, np.nan)
    _assert(positive_returns, 0, DAILY, np.nan)
    _assert(negative_returns, 0.0252, DAILY, -12.765908412673111)
    _assert(weekly_returns, 0, WEEKLY, 1.0520712810533321)
    _assert(weekly_returns, 0.052, WEEKLY, 0.8523637355083818)
    _assert(monthly_returns, 0, MONTHLY, 0.505398695719269)
    _assert(monthly_returns, 0.036, MONTHLY, 0.22425387870585353)


def test_tracking_error_information_ratio():
    def _assert(returns, benchmark, period, desired_te, desired_annual_te, desired_ir):
        r = _r(returns, pd.Series(benchmark.values, index=returns.index), 0, period)
        assert_almost_equal(r.tracking_error, desired_te)
        assert_almost_equal(r.annual_tracking_error, desired_annual_te)
        assert_almost_equal(r.information_ratio, desired_ir)

    _assert(positive_returns, zero_benchmark, DAILY, 0.0033333333333333335, 0.052915026221291815, 52.915026221291804)
    _assert(positive_returns, simple_benchmark, DAILY, 0.003333333333333333, 0.05291502622129181, 5.291502622129182)
    _assert(negative_returns, simple_benchmark, DAILY, 0.03179797338056485, 0.5047771785649584, -29.399110399937154)
    _assert(weekly_returns, simple_benchmark, WEEKLY, 0.05387743291748205, 0.38851569394870583, -0.5948565648975399)