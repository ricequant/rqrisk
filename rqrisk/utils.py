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

import numpy as np


class IndicatorProperty:
    pass


def indicator_property(min_period_count=None, value_when_pc_not_satisfied=np.nan):
    """
    封装绑定方法为缓存的 property 的装饰器
    :param min_period_count: 最小的 portfolio 长度，不满足时则给出 value_when_pc_not_satisfied，None 表示不做此项检查
    :param value_when_pc_not_satisfied: portfolio 长度小于 min_period_count 时给出的值，默认为 np.nan
    """

    class cached_property(IndicatorProperty):  # noqa
        def __init__(self, getter):
            if min_period_count is not None:
                self._getter = lambda i: value_when_pc_not_satisfied if i.period_count < min_period_count else getter(i)
            else:
                self._getter = getter
            self._name = getter.__name__

        def __get__(self, instance, owner):
            if instance is None:
                return self._getter
            value = self._getter(instance)
            setattr(instance, self._name, value)
            return value

    return cached_property


MONTHS_PER_YEAR = 12
WEEKS_PER_YEAR = 52
APPROX_BDAYS_PER_YEAR = 252
DAYS_PER_YEAR = 365

DAILY = 'daily'
WEEKLY = 'weekly'
MONTHLY = 'monthly'
YEARLY = 'yearly'
NATURAL_DAILY = "natural_daily"

ANNUALIZATION_FACTORS = {
    DAILY: APPROX_BDAYS_PER_YEAR,
    WEEKLY: WEEKS_PER_YEAR,
    MONTHLY: MONTHS_PER_YEAR,
    NATURAL_DAILY: DAYS_PER_YEAR
}


def annual_factor(period):
    try:
        return ANNUALIZATION_FACTORS[period]
    except KeyError:
        raise ValueError("period cannot be {}, possible values: {}".format(
            period, ", ".join(ANNUALIZATION_FACTORS.keys())))


def safe_div(dividend, divisor):
    if divisor == 0:
        return np.nan
    return dividend / divisor
