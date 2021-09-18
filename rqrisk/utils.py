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

DAILY = 'daily'
WEEKLY = 'weekly'
MONTHLY = 'monthly'
YEARLY = 'yearly'

ANNUALIZATION_FACTORS = {
    DAILY: APPROX_BDAYS_PER_YEAR,
    WEEKLY: WEEKS_PER_YEAR,
    MONTHLY: MONTHS_PER_YEAR
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
