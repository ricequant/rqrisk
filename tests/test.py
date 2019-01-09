#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rqdata
import rqrisk
from rqdata import *
from numpy.testing import assert_almost_equal
import pandas as pd
import numpy as np

#rqdata.init()
#
#def show_risk(r):
#    print('return           =', r.return_rate)
#    print('annual return    =', r.annual_return)
#    print('alpha            =', r.alpha)
#    print('beta             =', r.beta)
#    print('sharpe           =', r.sharpe)
#    print('benchmark return =', r.benchmark_return)
#    print('b annual_return  =', r.benchmark_annual_return)
#    print('max_drawdown     =', r.max_drawdown)
#    print('volatility       =', r.volatility)
#    print('annual volatility=', r.annual_volatility)
#    print('info. ratio      =', r.information_ratio)
#    print('downside risk    =', r.downside_risk)
#    print('sortino          =', r.sortino)
#    print('tracking error   =', r.tracking_error)
#    print('calmar           =', r.calmar)
#    print('VaR              =', r.var)
#    print('\n')
#
#
#p = industry('C27')
#pr = get_daily_returns(p, start_date='2012-01-01', end_date='2013-01-01')
#pr.fillna(0, inplace=True)
#pr = pr.sum(axis=1) / len(p)
#
#hs300 = index_components('000300.XSHG')
#br = get_daily_returns(hs300, start_date='2012-01-01', end_date='2013-01-01')
#br.fillna(0, inplace=True)
#br = br.sum(axis=1) / len(hs300)
#
#print('risk of pr, pr:')
#show_risk(rqrisk.Risk(pr, pr, 0))
#print('risk of br, br:')
#show_risk(rqrisk.Risk(br, br, 0))
#
#print('risk of pr, br:')
#show_risk(rqrisk.Risk(pr, br, 0))



# Simple benchmark, no drawdown
simple_benchmark = pd.Series(
    np.array([1., 1., 1., 1., 1., 1., 1., 1., 1.]) / 100,
    index=pd.date_range('2000-1-30', periods=9, freq='D'))

zero_benchmark = pd.Series(
    np.array([0]*9),
    index=pd.date_range('2000-1-30', periods=9, freq='D'))

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


def test_return():
    assert_almost_equal(
        rqrisk.Risk(positive_returns, simple_benchmark, 0).max_drawdown,
        -0.0)
    assert_almost_equal(
        rqrisk.Risk(negative_returns, simple_benchmark, 0).max_drawdown,
        -0.36590730349873601)
    assert_almost_equal(
        rqrisk.Risk(one_return, one_benchmark, 0).max_drawdown,
        0)


def test_annual_return():
    assert_almost_equal(
        rqrisk.Risk(weekly_returns, simple_benchmark, 0, rqrisk.WEEKLY).annual_return,
        0.24690830513998208)
    assert_almost_equal(
        rqrisk.Risk(monthly_returns, simple_benchmark, 0, rqrisk.MONTHLY).annual_return,
        0.052242061386048144)


def test_calmar():
    assert_almost_equal(
        rqrisk.Risk(one_return, one_benchmark, 0).calmar,
        np.nan)
    assert_almost_equal(
        rqrisk.Risk(weekly_returns, simple_benchmark, 0, rqrisk.WEEKLY).calmar,
        2.4690830513998208)
    assert_almost_equal(
        rqrisk.Risk(monthly_returns, simple_benchmark, 0, rqrisk.MONTHLY).calmar,
        0.52242061386048144)


def test_annual_volatity():
    assert_almost_equal(
        rqrisk.Risk(simple_benchmark, simple_benchmark, 0).annual_volatility,
        0)
    assert_almost_equal(
        rqrisk.Risk(weekly_returns, simple_benchmark, 0, rqrisk.WEEKLY).annual_volatility,
        0.38851569394870583)
    assert_almost_equal(
        rqrisk.Risk(monthly_returns, simple_benchmark, 0, rqrisk.MONTHLY).annual_volatility,
        0.18663690238892558)


def test_sharpe():
    assert_almost_equal(
        rqrisk.Risk(one_return, one_benchmark, 0).sharpe,
        np.nan)
    assert_almost_equal(
        rqrisk.Risk(positive_returns, zero_benchmark, 0).sharpe,
        52.915026221291804)
    assert_almost_equal(
        rqrisk.Risk(negative_returns, zero_benchmark, 0).sharpe,
        -24.406808633910085)


def test_downside_risk():
    assert_almost_equal(
        rqrisk.Risk(one_return, np.array([0]), 0).downside_risk,
        0.0)
    assert_almost_equal(
        rqrisk.Risk(weekly_returns,
                    pd.Series(zero_benchmark.values, index=weekly_returns.index),
                    0, rqrisk.WEEKLY).downside_risk,
        0.25888650451930134)
    assert_almost_equal(
        rqrisk.Risk(weekly_returns,
                    pd.Series(dot_one_benchmark, index=weekly_returns.index),
                    0, rqrisk.WEEKLY).downside_risk,
        0.7733045971672482)
    assert_almost_equal(
        rqrisk.Risk(monthly_returns,
                    pd.Series(zero_benchmark.values, index=monthly_returns.index),
                    0, rqrisk.MONTHLY).downside_risk,
        0.1243650540411842)
    assert_almost_equal(
        rqrisk.Risk(monthly_returns,
                    pd.Series(dot_one_benchmark, index=monthly_returns.index),
                    0, rqrisk.MONTHLY).downside_risk,
        0.37148351242013422)


def test_sortino():
    assert_almost_equal(
        rqrisk.Risk(one_return, one_benchmark, 0).sortino,
        np.nan)
    assert_almost_equal(
        rqrisk.Risk(positive_returns,
                    pd.Series(zero_benchmark.values, index=positive_returns.index),
                    0).sortino,
        np.inf)
    assert_almost_equal(
        rqrisk.Risk(negative_returns,
                    pd.Series(zero_benchmark.values, index=negative_returns.index),
                    0).sortino,
        -13.532743075043401)
    assert_almost_equal(
        rqrisk.Risk(simple_benchmark,
                    pd.Series(zero_benchmark.values, index=simple_benchmark.index),
                    0).sortino,
        np.inf)
    assert_almost_equal(
        rqrisk.Risk(weekly_returns,
                    pd.Series(zero_benchmark.values, index=weekly_returns.index),
                    0, rqrisk.WEEKLY).sortino,
        0.50690062680370862)
    assert_almost_equal(
        rqrisk.Risk(monthly_returns,
                    pd.Series(zero_benchmark.values, index=monthly_returns.index),
                    0, rqrisk.MONTHLY).sortino,
        0.11697706772393276)


def test_information_ratio():
    assert_almost_equal(
        rqrisk.Risk(positive_returns,
                    pd.Series(zero_benchmark.values, index=positive_returns.index),
                    0).information_ratio,
        3.3333333333333326)
    assert_almost_equal(
        rqrisk.Risk(negative_returns,
                    pd.Series(zero_benchmark.values, index=negative_returns.index),
                    0).information_ratio,
        -1.5374844271921471)

