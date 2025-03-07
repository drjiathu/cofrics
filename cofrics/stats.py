from __future__ import annotations, division

import math
from math import pow
from sys import float_info

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .periods import (
    ANNUALIZATION_FACTORS,
    APPROX_BDAYS_PER_YEAR,
    DAILY,
    MONTHLY,
    QUARTERLY,
    WEEKLY,
    YEARLY,
)
from .utils import down, nanmean, nanmin, nanstd, roll, rolling_window, up

DEFAULT_THRESHOLD = 0.2
MINIMUM_THRESHOLD = 1e-10


def _create_unary_vectorized_roll_function(function):
    def unary_vectorized_roll(arr, window, out=None, **kwargs):
        """Computes the {human_readable} measure over a rolling window.

        Parameters
        ----------
        arr: array-like
            The array to compute the rolling {human_readable} over.

        window: int
            Size of the rolling window in terms of the periodicity of the data.

        out: array-like, optional
            Array to use as output buffer.
            If not passed, a new array will be created.

        **kwargs
            Forwarded to :func:`~.{name}`.

        Returns
        -------
        rolling_{name}: array-like
            The rolling {human_readable}.
        """
        allocated_output = out is None

        if len(arr):
            out = function(
                rolling_window(_flatten(arr), min(len(arr), window)).T,
                out=out,
                **kwargs,
            )
        else:
            out = np.empty(0, dtype="float64")

        if allocated_output and isinstance(arr, pd.Series):
            out = pd.Series(out, index=arr.index[-len(out) :])

        return out

    unary_vectorized_roll.__doc__ = unary_vectorized_roll.__doc__.format(
        name=function.__name__,
        human_readable=function.__name__.replace("_", " "),
    )

    return unary_vectorized_roll


def _create_binary_vectorized_roll_function(function):
    def binary_vectorized_roll(lhs, rhs, window, out=None, **kwargs):
        """Computes the {human_readable} measure over a rolling window.

        Parameters
        ----------
        lhs: array-like
            The first array to pass to the rolling {human_readable}.

        rhs: array-like
            The second array to pass to the rolling {human_readable}.

        window: int
            Size of the rolling window in terms of the periodicity of the data.

        out: array-like, optional
            Array to use as output buffer.
            If not passed, a new array will be created.

        **kwargs
            Forwarded to :func:`~.{name}`.

        Returns
        -------
        rolling_{name}: array-like
            The rolling {human_readable}.
        """
        allocated_output = out is None

        if window >= 1 and len(lhs) and len(rhs):
            out = function(
                rolling_window(_flatten(lhs), min(len(lhs), window)).T,
                rolling_window(_flatten(rhs), min(len(rhs), window)).T,
                out=out,
                **kwargs,
            )
        elif allocated_output:
            out = np.empty(0, dtype="float64")
        else:
            out[()] = np.nan

        if allocated_output:
            if out.ndim == 1 and isinstance(lhs, pd.Series):
                out = pd.Series(out, index=lhs.index[-len(out) :])
            elif out.ndim == 2 and isinstance(lhs, pd.Series):
                out = pd.DataFrame(out, index=lhs.index[-len(out) :])
        return out

    binary_vectorized_roll.__doc__ = binary_vectorized_roll.__doc__.format(
        name=function.__name__,
        human_readable=function.__name__.replace("_", " "),
    )

    return binary_vectorized_roll


def _flatten(arr):
    """Turn a pd.Series or an array-like object to an array.

    Parameters
    ----------
    arr: array-like | pd.Series

    Returns
    -------
    array-like
    """
    return arr.values if isinstance(arr, pd.Series) else arr


def _adjust_returns(
    returns: pd.Series | np.ndarray,
    adjustment_factor: pd.Series | np.ndarray | float | int,
) -> np.ndarray | pd.Series:
    """Returns the return series adjusted by adjustment_factor.

    Optimizes for the case of adjustment_factor being 0 by returning `returns`
    itself, not a copy!

    Parameters
    ----------
    returns: pd.Series | np.ndarray

    adjustment_factor: pd.Series | np.ndarray | float | int

    Returns
    -------
    adjusted_returns: array-like
    """
    if isinstance(adjustment_factor, (float, int)) and adjustment_factor == 0:
        return returns

    return returns - adjustment_factor


def annualization_factor(period: str, annualization=None) -> int | float:
    """Return annualization factor from period entered or if a custom value is
    passed in.

    Parameters
    ----------
    period: str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'monthly': 12
            'weekly': 52
            'daily' : 252

    annualization: int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.

    Returns
    -------
    annualization_factor: int | float
    """
    if annualization is None:
        try:
            factor = ANNUALIZATION_FACTORS[period]
        except KeyError:
            raise ValueError(
                "Period cannot be '{}'. Can be '{}'.".format(
                    period, "', '".join(ANNUALIZATION_FACTORS.keys())
                )
            )
    else:
        factor = annualization

    return factor


def simple_returns(prices: pd.Series | pd.DataFrame | np.ndarray):
    """Compute simple returns from a time-series of prices.

    Parameters
    ----------
    prices: pd.Series | pd.DataFrame | np.ndarray
        Prices of assets in wide-format, with assets as columns and indexed by
        datetime.

    Returns
    -------
    returns: array-like
        Returns of assets in wide-format, with assets as columns, and index
        coerced to be tz-aware.
    """
    if isinstance(prices, (pd.DataFrame, pd.Series)):
        out = prices.pct_change().iloc[1:]
    else:
        # Assume np.ndarray
        out = np.diff(prices, axis=0)
        np.divide(out, prices[:-1], out=out)

    return out


def cum_returns(
    returns: pd.Series | pd.DataFrame | np.ndarray,
    starting_value: float = 0.0,
    out=None,
):
    """Compute cumulative returns from simple returns.

    Parameters
    ----------
    returns: pd.Series | np.ndarray | pd.DataFrame
        Returns of the strategy as a percentage, noncumulative.
         - Time series with decimal returns.
         - Example::

            2015-07-16   -0.012143
            2015-07-17    0.045350
            2015-07-20    0.030957
            2015-07-21    0.004902

         - Also accepts two-dimensional data. In this case, each column is
           cumulated.

    starting_value: float, optional
       The starting returns.

    out: array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    cumulative_returns: array-like
        Series of cumulative returns.
    """
    if len(returns) < 1:
        return returns.copy()

    nanmask = np.isnan(returns)
    if np.any(nanmask):
        returns = returns.copy()
        returns[nanmask] = 0.0

    allocated_output = out is None
    if allocated_output:
        out = np.empty_like(returns)

    np.add(returns, 1, out=out)
    out.cumprod(axis=0, out=out)

    if starting_value == 0:
        np.subtract(out, 1, out=out)
    elif starting_value > 0:
        np.multiply(out, starting_value, out=out)
    else:
        raise ValueError(
            f"The starting value to calculate cumulative return should be non-negative,"
            f" but got {starting_value} instead."
        )

    if allocated_output:
        if returns.ndim == 1 and isinstance(returns, pd.Series):
            out = pd.Series(out, index=returns.index)
        elif isinstance(returns, pd.DataFrame):
            out = pd.DataFrame(out, index=returns.index, columns=returns.columns)

    return out


def cum_returns_final(
    returns: pd.DataFrame | pd.Series | np.ndarray, starting_value: float = 0.0
):
    """Compute total returns from simple returns.

    Parameters
    ----------
    returns: pd.DataFrame | pd.Series, | np.ndarray
       Noncumulative simple returns of one or more time-series.

    starting_value: float, optional
       The starting returns.

    Returns
    -------
    total_returns: pd.Series | np.ndarray | float
        If input is 1-dimensional (a Series or 1D numpy array), the result is a scalar.
        If input is 2-dimensional (a DataFrame or 2D numpy array), the result is a
        1D-array containing cumulative returns for each column of input.
    """
    if len(returns) == 0:
        return np.nan

    result = (
        (returns + 1).prod()
        if isinstance(returns, pd.DataFrame)
        else np.nanprod(returns + 1, axis=0)
    )

    if starting_value == 0:
        result -= 1
    elif starting_value > 0:
        result *= starting_value
    else:
        raise ValueError(
            f"The starting value to calculate cumulative return should be non-negative,"
            f" but got {starting_value} instead."
        )

    return result


def aggregate_returns(returns: pd.Series, convert_to: str) -> pd.Series:
    """Aggregates returns by week, month, or year.

    Parameters
    ----------
    returns: pd.Series
       Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~.stats.cum_returns`.

    convert_to: str
        Can be 'weekly', 'monthly', or 'yearly'.

    Returns
    -------
    aggregated_returns: pd.Series
    """

    def cumulate_returns(x):
        cum_ret = cum_returns(x)
        return cum_ret.iloc[-1] if isinstance(cum_ret, pd.Series) else cum_ret[-1]

    if convert_to == WEEKLY:
        grouping = [lambda x: x.year, lambda x: x.isocalendar()[1]]
    elif convert_to == MONTHLY:
        grouping = [lambda x: x.year, lambda x: x.month]
    elif convert_to == QUARTERLY:
        grouping = [lambda x: x.year, lambda x: int(math.ceil(x.month / 3.0))]
    elif convert_to == YEARLY:
        grouping = [lambda x: x.year]
    else:
        raise ValueError(f"convert_to must be {WEEKLY}, {MONTHLY} or {YEARLY}")

    return returns.groupby(grouping).apply(cumulate_returns)


def max_drawdown(returns: pd.Series | np.ndarray, out=None) -> float | pd.Series:
    """Determines the maximum drawdown of a strategy.

    Parameters
    ----------
    returns: pd.Series | np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~.stats.cum_returns`.
    out: array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    max_drawdown: float

    References
    ----------
    See https://en.wikipedia.org/wiki/Drawdown_(economics) for more details.
    """
    allocated_output = out is None
    if allocated_output:
        out = np.empty(returns.shape[1:])

    returns_1d = returns.ndim == 1

    if len(returns) < 1:
        out[()] = np.nan
        if returns_1d:
            out = out.item()
        return out

    returns_array = np.asanyarray(returns)

    cumulative = np.empty(
        shape=(returns.shape[0] + 1,) + returns.shape[1:], dtype="float64"
    )
    cumulative[0] = start = 100
    cum_returns(returns_array, starting_value=start, out=cumulative[1:])

    max_return = np.fmax.accumulate(cumulative, axis=0)

    nanmin((cumulative - max_return) / max_return, axis=0, out=out)

    if returns_1d:
        out = out.item()
    elif allocated_output and isinstance(returns, pd.DataFrame):
        out = pd.Series(out)

    return out


roll_max_drawdown = _create_unary_vectorized_roll_function(max_drawdown)


def annual_return(
    returns: pd.Series | np.ndarray, period=DAILY, annualization=None
) -> float:
    """Determines the mean annual growth rate of returns.

    This is equivalent to the compound annual growth rate.

    Parameters
    ----------
    returns: pd.Series | np.ndarray
        Periodic returns of the strategy, noncumulative.
        - See full explanation in :func:`~.stats.cum_returns`.

    period: str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'monthly':12
            'weekly': 52
            'daily': 252

    annualization: int | float, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.

    Returns
    -------
    annual_return: float
        Annual Return as CAGR (Compounded Annual Growth Rate).
    """

    if len(returns) < 1:
        return np.nan

    ann_factor = annualization_factor(period, annualization)
    num_years = len(returns) / ann_factor
    # Pass array to ensure index -1 looks up successfully.
    ending_value = cum_returns_final(returns, starting_value=1)

    return ending_value ** (1 / num_years) - 1


def cagr(returns, period=DAILY, annualization=None):
    """Compute compound annual growth rate. Alias function for
    :func:`~.stats.annual_return`

    Parameters
    ----------
    returns: pd.Series | np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~.stats.cum_returns`.

    period: str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'monthly':12
            'weekly': 52
            'daily': 252

    annualization: int | float, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
        - See full explanation in :func:`~.stats.annual_return`.

    Returns
    -------
    cagr: float
        The CAGR value.
    """

    return annual_return(returns, period, annualization)


roll_cagr = _create_unary_vectorized_roll_function(cagr)


def annual_volatility(
    returns: pd.Series | pd.DataFrame,
    period=DAILY,
    alpha: float = 2.0,
    annualization=None,
    out=None,
) -> float:
    """Determine the annual volatility of a strategy.

    Parameters
    ----------
    returns: pd.Series | np.ndarray
        Periodic returns of the strategy, noncumulative.
        - See full explanation in :func:`~.stats.cum_returns`.

    period: str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'monthly':12
            'weekly': 52
            'daily': 252

    alpha: float, optional
        Scaling relation (Levy stability exponent).

    annualization: int | float, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.

    out: array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    annual_volatility: float

    """
    allocated_output = out is None
    if allocated_output:
        out = np.empty(returns.shape[1:])

    returns_1d = returns.ndim == 1

    if len(returns) < 2:
        out[()] = np.nan
        if returns_1d:
            out = out.item()

        return out

    ann_factor = annualization_factor(period, annualization)
    nanstd(returns, ddof=1, axis=0, out=out)
    out = np.multiply(out, ann_factor ** (1.0 / alpha), out)

    if returns_1d:
        out = out.item()

    return out


roll_annual_volatility = _create_unary_vectorized_roll_function(annual_volatility)


def calmar_ratio(returns, period=DAILY, annualization=None):
    """Determines the Calmar ratio, or drawdown ratio, of a strategy.

    Parameters
    ----------
    returns: pd.Series | np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~cofrics.stats.cum_returns`.

    period: str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'monthly':12
            'weekly': 52
            'daily': 252

    annualization: int | float, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.

    Returns
    -------
    calmar_ratio: float
        Calmar ratio (drawdown ratio) as float. Returns a NaN if there is no
        calmar ratio.

    References
    ----------
    See https://en.wikipedia.org/wiki/Calmar_ratio for more details.
    """

    max_dd = max_drawdown(returns=returns)
    max_dd = max_dd.item() if isinstance(max_dd, pd.Series) else max_dd
    if max_dd < 0:
        temp = annual_return(returns, period, annualization) / abs(max_dd)
    else:
        return np.nan

    return np.nan if np.isinf(temp) else temp


def omega_ratio(
    returns,
    risk_free=0,
    required_return: float = 0.0,
    annualization=APPROX_BDAYS_PER_YEAR,
):
    """Determines the Omega ratio of a strategy.

    Parameters
    ----------
    returns: pd.Series | np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~cofrics.stats.cum_returns`.

    risk_free: int | float
        Constant risk-free return throughout the period

    required_return: float, optional
        Minimum acceptance return of the investor. Threshold over which to
        consider positive vs negative returns. It will be converted to a
        value appropriate for the period of the returns.
        E.g. An annual minimum acceptable return of 100 will translate to a
        minimum acceptable return of 0.018.

    annualization: int | float, optional
        Factor used to convert the required_return into a daily value.
        Enter 1 if no time period conversion is necessary.

    Returns
    -------
    omega_ratio: float

    References
    ----------
    See https://en.wikipedia.org/wiki/Omega_ratio for more details.

    """

    if len(returns) < 2:
        return np.nan

    if annualization == 1:
        return_threshold = required_return
    elif required_return <= -1:
        return np.nan
    else:
        return_threshold = (1 + required_return) ** (1.0 / annualization) - 1

    returns_less_thresh = returns - risk_free - return_threshold

    numerator = sum(returns_less_thresh[returns_less_thresh > 0.0])
    denominator = -1.0 * sum(returns_less_thresh[returns_less_thresh < 0.0])

    return numerator / denominator if denominator > 0 else np.nan


def sharpe_ratio(
    returns: pd.Series | np.ndarray,
    risk_free: int | float = 0.0,
    period: str = DAILY,
    annualization: int = None,
    out=None,
):
    """Determines the Sharpe ratio of a strategy.

    Parameters
    ----------
    returns: pd.Series | np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~cofrics.stats.cum_returns`.

    risk_free: int | float
        Constant daily risk-free return throughout the period.

    period: str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'monthly':12
            'weekly': 52
            'daily': 252

    annualization: int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.

    out: array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    sharpe_ratio: float
        NaN if insufficient length of returns or if adjusted returns are 0.

    References
    ----------
    See https://en.wikipedia.org/wiki/Sharpe_ratio for more details.

    """

    allocated_output = out is None
    if allocated_output:
        out = np.empty(returns.shape[1:])

    return_1d = returns.ndim == 1

    if len(returns) < 2:
        out[()] = np.nan
        if return_1d:
            out = out.item()
        return out

    returns_risk_adj = np.asanyarray(_adjust_returns(returns, risk_free))
    ann_factor = annualization_factor(period, annualization)

    np.multiply(
        np.divide(
            nanmean(returns_risk_adj, axis=0),
            nanstd(returns_risk_adj, ddof=1, axis=0),
            out=out,
        ),
        np.sqrt(ann_factor),
        out=out,
    )

    if return_1d:
        out = out.item()

    return out


roll_sharpe_ratio = _create_unary_vectorized_roll_function(sharpe_ratio)


def sortino_ratio(
    returns: pd.Series | pd.DataFrame | np.ndarray,
    required_return: float | pd.Series = 0.0,
    period: str = DAILY,
    annualization: int | float = None,
    out=None,
    _downside_risk: float = None,
) -> float | pd.Series:
    """Determines the Sortino ratio of a strategy.

    Parameters
    ----------
    returns: pd.Series | np.ndarray | pd.DataFrame
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~cofrics.stats.cum_returns`.

    required_return: float | pd.Series
        minimum acceptable return

    period: str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'monthly':12
            'weekly': 52
            'daily': 252

    annualization: int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.

    out: array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    _downside_risk: float, optional
        The downside risk of the given inputs, if known. Will be calculated if
        not provided.

    Returns
    -------
    sortino_ratio: float | pd.Series

        depends on input type
        series ==> float
        DataFrame ==> pd.Series

    References
    ----------
    See https://en.wikipedia.org/wiki/Sortino_ratio for more details.

    """
    allocated_output = out is None
    if allocated_output:
        out = np.empty(returns.shape[1:])

    return_1d = returns.ndim == 1

    if len(returns) < 2:
        out[()] = np.nan
        if return_1d:
            out = out.item()
        return out

    adj_returns = np.asanyarray(_adjust_returns(returns, required_return))

    ann_factor = annualization_factor(period, annualization)

    average_annual_return = nanmean(adj_returns, axis=0) * ann_factor
    annualized_downside_risk = (
        _downside_risk
        if _downside_risk is not None
        else downside_risk(returns, required_return, period, annualization)
    )
    np.divide(average_annual_return, annualized_downside_risk, out=out)

    if return_1d:
        out = out.item()
    elif isinstance(returns, pd.DataFrame):
        out = pd.Series(out)

    return out


roll_sortino_ratio = _create_unary_vectorized_roll_function(sortino_ratio)


def downside_risk(
    returns: pd.Series | pd.DataFrame | np.ndarray,
    required_return: float | pd.Series = 0.0,
    period: str = DAILY,
    annualization: int | float = None,
    out=None,
) -> float | pd.Series:
    """Determines the downside deviation below a threshold.

    Parameters
    ----------
    returns: pd.Series | np.ndarray | pd.DataFrame
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~cofrics.stats.cum_returns`.

    required_return: float | pd.Series
        minimum acceptable return

    period: str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'monthly':12
            'weekly': 52
            'daily': 252

    annualization: int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.

    out: array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    downside_deviation: float | pd.Series
        depends on input type
        series ==> float
        DataFrame ==> pd.Series
    """
    allocated_output = out is None
    if allocated_output:
        out = np.empty(returns.shape[1:])

    returns_1d = returns.ndim == 1

    if len(returns) < 1:
        out[()] = np.nan
        if returns_1d:
            out = out.item()
        return out

    ann_factor = annualization_factor(period, annualization)

    downside_diff = np.clip(
        _adjust_returns(
            np.asanyarray(returns),
            np.asanyarray(required_return),
        ),
        -np.inf,
        0,
    )

    np.square(downside_diff, out=downside_diff)
    nanmean(downside_diff, axis=0, out=out)
    np.sqrt(out, out=out)
    np.multiply(out, np.sqrt(ann_factor), out=out)

    if returns_1d:
        out = out.item()
    elif isinstance(returns, pd.DataFrame):
        out = pd.Series(out, index=returns.columns)

    return out


roll_downsize_risk = _create_unary_vectorized_roll_function(downside_risk)


def excess_sharpe(
    returns: pd.Series | np.ndarray, factor_returns: float | pd.Series, out=None
) -> float:
    """Determines the Excess Sharpe of a strategy.

    Parameters
    ----------
    returns: pd.Series | np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~cofrics.stats.cum_returns`.

    factor_returns: float | pd.Series
        Benchmark return to compare returns against.

    out: array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    excess_sharpe: float

    References
    ----------
    The excess Sharpe is a simplified Information Ratio that uses tracking
    error rather than "active risk" as the denominator.
    """
    allocated_output = out is None
    if allocated_output:
        out = np.empty(returns.shape[1:])

    returns_1d = returns.ndim == 1

    if len(returns) < 2:
        out[()] = np.nan
        if returns_1d:
            out = out.item()
        return out

    active_return = _adjust_returns(returns, factor_returns)
    tracking_error = np.nan_to_num(nanstd(active_return, ddof=1, axis=0))

    out = np.divide(
        nanmean(active_return, axis=0, out=out),
        tracking_error,
        out=out,
    )
    if returns_1d:
        out = out.item()
    return out


roll_excess_sharpe = _create_binary_vectorized_roll_function(excess_sharpe)


def _to_pandas(ob) -> pd.Series | pd.DataFrame:
    """Convert an array-like to a pandas object.

    Parameters
    ----------
    ob: array-like
        The object to convert.

    Returns
    -------
    pandas_structure: pd.Series | pd.DataFrame
        The correct structure based on the dimensionality of the data.
    """
    if isinstance(ob, (pd.Series, pd.DataFrame)):
        return ob

    if ob.ndim == 1:
        return pd.Series(ob)
    elif ob.ndim == 2:
        return pd.DataFrame(ob)
    else:
        raise ValueError(
            "cannot convert array of dim > 2 to a pandas Series or DataFrame structure"
        )


def _aligned_series(*many_series):
    """Return a new list of series containing the data in the input series, but
    with their indices aligned.

    NaNs will be filled in for missing values.

    Parameters
    ----------
    *many_series
        The series to align.

    Returns
    -------
    aligned_series: iterable[array-like]
        A new list of series containing the data in the input series, but
        with their indices aligned. NaNs will be filled in for missing values.

    """
    head, tails = many_series[0], many_series[1:]
    n = len(head)
    if isinstance(head, np.ndarray) and all(
        len(s) == n and isinstance(s, np.ndarray) for s in tails
    ):
        # Optimization: ndarray of the same length are already aligned
        return many_series

    # Dataframe has no ``itervalues``
    return (v for _, v in pd.concat(map(_to_pandas, many_series), axis=1).items())


def alpha_beta(
    returns: pd.Series | np.array,
    factor_returns: pd.Series | np.array,
    risk_free: int | float = 0.0,
    period=DAILY,
    annualization=None,
    out=None,
) -> tuple[float, float]:
    """Calculates annualized alpha and beta.

    Parameters
    ----------
    returns: pd.Series | np.array
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~cofrics.stats.cum_returns`.

    factor_returns: pd.Series | np.array
         Daily noncumulative returns of the factor to which beta is computed.
         Usually a benchmark such as the market.
         - This is in the same style as returns.

    risk_free: int | float, optional
        Constant risk-free return throughout the period. For example, the
        interest rate on a three-month US treasury bill.

    period: str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'monthly':12
            'weekly': 52
            'daily': 252

    annualization: int | float, optional
        Used to suppress default values available in `period` to convert
        `returns` into annual returns. Value should be the annual frequency of
        `returns`.

    out: array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    alpha: float
    beta: float
    """
    returns, factor_returns = _aligned_series(returns, factor_returns)

    return alpha_beta_aligned(
        returns,
        factor_returns,
        risk_free=risk_free,
        period=period,
        annualization=annualization,
        out=out,
    )


def roll_alpha_beta(returns, factor_returns, window=10, **kwargs):
    """Computes alpha and beta over a rolling window.

    Parameters
    ----------
    returns: array-like
        The first array to pass to the rolling alpha-beta.

    factor_returns: array-like
        The second array to pass to the rolling alpha-beta.

    window: int
        Size of the rolling window in terms of the periodicity of the data.

    out (keyword): array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    **kwargs
        Forwarded to :func:`~cofrics.alpha_beta`.
    """
    returns, factor_returns = _aligned_series(returns, factor_returns)

    return roll_alpha_beta_aligned(returns, factor_returns, window=window, **kwargs)


def alpha_beta_aligned(
    returns: pd.Series | np.ndarray,
    factor_returns: pd.Series | np.ndarray,
    risk_free: int | float = 0,
    period: str = DAILY,
    annualization: int = None,
    out=None,
) -> tuple[float, float]:
    """Calculates annualized alpha and beta.

    If they are pd.Series, expects returns and factor_returns have already
    been aligned on their labels.  If np.ndarray, these arguments should have
    the same shape.

    Parameters
    ----------
    returns: pd.Series | np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~cofrics.stats.cum_returns`.

    factor_returns: pd.Series | np.ndarray
         Daily noncumulative returns of the factor to which beta is
         computed. Usually a benchmark such as the market.
         - This is in the same style as returns.

    risk_free: int | float, optional
        Constant risk-free return throughout the period. For example, the
        interest rate on a three month us treasury bill.

    period: str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'monthly':12
            'weekly': 52
            'daily': 252

    annualization: int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.

    out: array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    alpha: float
    beta: float
    """
    if out is None:
        out = np.empty(returns.shape[1:] + (2,), dtype="float64")

    b = beta_aligned(returns, factor_returns, risk_free, out=out[..., 1])
    alpha_aligned(
        returns,
        factor_returns,
        risk_free,
        period,
        annualization,
        out=out[..., 0],
        _beta=b,
    )

    return out


roll_alpha_beta_aligned = _create_binary_vectorized_roll_function(alpha_beta_aligned)


def alpha(
    returns: pd.Series,
    factor_returns: pd.Series,
    risk_free: int | float = 0.0,
    period=DAILY,
    annualization=None,
    out=None,
    _beta=None,
) -> float:
    """Calculates annualized alpha.

    Parameters
    ----------
    returns: pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~cofrics.stats.cum_returns`.

    factor_returns: pd.Series
        Daily noncumulative returns of the factor to which beta is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.

    risk_free: int | float, optional
        Constant risk-free return throughout the period. For example, the
        interest rate on a three month us treasury bill.

    period: str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'monthly':12
            'weekly': 52
            'daily': 252

    annualization: int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
        - See full explanation in :func:`~cofrics.stats.annual_return`.

    out: array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    _beta: float, optional
        The beta for the given inputs, if already known. Will be calculated
        internally if not provided.

    Returns
    -------
    alpha: float
    """
    if not (isinstance(returns, np.ndarray) and isinstance(factor_returns, np.ndarray)):
        returns, factor_returns = _aligned_series(returns, factor_returns)

    return alpha_aligned(
        returns,
        factor_returns,
        risk_free=risk_free,
        period=period,
        annualization=annualization,
        out=out,
        _beta=_beta,
    )


roll_alpha = _create_binary_vectorized_roll_function(alpha)


def alpha_aligned(
    returns: pd.Series | np.ndarray,
    factor_returns: pd.Series | np.ndarray,
    risk_free: int | float = 0,
    period: str = DAILY,
    annualization: int = None,
    out=None,
    _beta: float = None,
) -> float:
    """Calculates annualized alpha.

    If they are pd.Series, expects returns and factor_returns have already
    been aligned on their labels.  If np.ndarray, these arguments should have
    the same shape.

    Parameters
    ----------
    returns: pd.Series | np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~cofrics.stats.cum_returns`.

    factor_returns: pd.Series | np.ndarray
        Daily noncumulative returns of the factor to which beta is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.

    risk_free: int | float, optional
        Constant risk-free return throughout the period. For example, the
        interest rate on a three month us treasury bill.

    period: str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'monthly':12
            'weekly': 52
            'daily': 252

    annualization: int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
        - See full explanation in :func:`~cofrics.stats.annual_return`.

    out: array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    _beta: float, optional
        The beta for the given inputs, if already known. Will be calculated
        internally if not provided.

    Returns
    -------
    alpha: float
    """
    allocated_output = out is None
    if allocated_output:
        out = np.empty(returns.shape[1:], dtype="float64")

    if len(returns) < 2:
        out[()] = np.nan
        if returns.ndim == 1:
            out = out.item()
        return out

    ann_factor = annualization_factor(period, annualization)

    if _beta is None:
        _beta = beta_aligned(returns, factor_returns, risk_free)

    adj_returns = _adjust_returns(returns, risk_free)
    adj_factor_returns = _adjust_returns(factor_returns, risk_free)
    alpha_series = adj_returns - (_beta * adj_factor_returns)

    out = np.subtract(
        np.power(
            np.add(nanmean(alpha_series, axis=0, out=out), 1, out=out),
            ann_factor,
            out=out,
        ),
        1,
        out=out,
    )

    if allocated_output and isinstance(returns, pd.DataFrame):
        out = pd.Series(out)

    if returns.ndim == 1:
        out = out.item()

    return out


roll_alpha_aligned = _create_binary_vectorized_roll_function(alpha_aligned)


def beta(
    returns: pd.Series, factor_returns: pd.Series, risk_free: int | float = 0, out=None
) -> float:
    """Calculates beta.

    Parameters
    ----------
    returns: pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~cofrics.stats.cum_returns`.

    factor_returns: pd.Series
         Daily noncumulative returns of the factor to which beta is
         computed. Usually a benchmark such as the market.
         - This is in the same style as returns.

    risk_free: int | float, optional
        Constant risk-free return throughout the period. For example, the
        interest rate on a three month us treasury bill.

    out: array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    beta: float
    """
    if not (isinstance(returns, np.ndarray) and isinstance(factor_returns, np.ndarray)):
        returns, factor_returns = _aligned_series(returns, factor_returns)

    return beta_aligned(returns, factor_returns, risk_free=risk_free, out=out)


roll_beta = _create_binary_vectorized_roll_function(beta)


def beta_aligned(
    returns: pd.Series | np.ndarray,
    factor_returns: pd.Series | np.ndarray,
    risk_free: int | float = 0.0,
    out=None,
) -> float:
    """Calculates beta.

    If they are pd.Series, expects returns and factor_returns have already
    been aligned on their labels.  If np.ndarray, these arguments should have
    the same shape.

    Parameters
    ----------
    returns: pd.Series | np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~cofrics.stats.cum_returns`.

    factor_returns: pd.Series | np.ndarray
         Daily noncumulative returns of the factor to which beta is
         computed. Usually a benchmark such as the market.
         - This is in the same style as returns.

    risk_free: int | float, optional
        Constant risk-free return throughout the period. For example, the
        interest rate on a three month us treasury bill.

    out: array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    beta: float
    """
    # Cache these as locals since we're going to call them multiple times.
    nan, isnan = np.nan, np.isnan
    returns_1d = returns.ndim == 1

    if returns_1d:
        returns = np.asanyarray(returns)[:, np.newaxis]

    if factor_returns.ndim == 1:
        factor_returns = np.asanyarray(factor_returns)[:, np.newaxis]

    m = returns.shape[1]

    if out is None:
        out = np.full(m, nan)
    elif out.ndim == 0:
        out = out[np.newaxis]

    if len(returns) < 1 or len(factor_returns) < 2:
        out[()] = nan
        if returns_1d:
            out = out.item()
        return out

    # Copy N times as a column vector and fill with nans to have the same
    # missing value pattern as the dependent variable.
    #
    # TODO: [PERF] We could probably avoid the space blowup by doing this in
    #  Cython.

    # shape: (N, M)
    independent = np.where(isnan(returns), nan, factor_returns)

    # Calculate beta as Cov(X, Y) / Cov(X, X).
    # https://en.wikipedia.org/wiki/Simple_linear_regression#Fitting_the_regression_line  # noqa
    #
    # NOTE: The usual formula for covariance is::
    #
    #    mean((X - mean(X)) * (Y - mean(Y)))
    #
    # However, we don't actually need to take the mean of both sides of the
    # product, because of the following equivalence::
    #
    # Let X_res = (X - mean(X)).
    # We have:
    #
    #     mean(X_res * (Y - mean(Y))) = mean(X_res * (Y - mean(Y)))
    #                             (1) = mean((X_res * Y) - (X_res * mean(Y)))
    #                             (2) = mean(X_res * Y) - mean(X_res * mean(Y))
    #                             (3) = mean(X_res * Y) - mean(X_res) * mean(Y)
    #                             (4) = mean(X_res * Y) - 0 * mean(Y)
    #                             (5) = mean(X_res * Y)
    #
    #
    # The tricky step in the above derivation is step (4). We know that
    # mean(X_res) is zero because, for any X:
    #
    #     mean(X - mean(X)) = mean(X) - mean(X) = 0.
    #
    # The upshot of this is that we only have to center one of `independent`
    # and `dependent` when calculating covariances. Since we need the centered
    # `independent` to calculate its variance in the next step, we choose to
    # center `independent`.

    ind_residual = independent - nanmean(independent, axis=0)

    covariances = nanmean(ind_residual * returns, axis=0)

    # We end up with different variances in each column here because each
    # column may have a different subset of the data dropped due to missing
    # data in the corresponding dependent column.
    # shape: (M,)
    np.square(ind_residual, out=ind_residual)
    independent_variances = nanmean(ind_residual, axis=0)
    independent_variances[independent_variances < 1.0e-30] = np.nan

    np.divide(covariances, independent_variances, out=out)

    if returns_1d:
        out = out.item()

    return out


roll_beta_aligned = _create_binary_vectorized_roll_function(beta_aligned)


def stability_of_timeseries(returns: pd.Series | np.ndarray) -> float:
    """Determines R-squared of a linear fit to the cumulative log returns.

    Computes an ordinary least squares linear fit, and returns R-squared.

    Parameters
    ----------
    returns: pd.Series | np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~cofrics.stats.cum_returns`.

    Returns
    -------
    R-squared: float
    """
    if len(returns) < 2:
        return np.nan

    returns = np.asanyarray(returns)
    returns = returns[~np.isnan(returns)]

    cum_log_returns = np.log1p(returns).cumsum()

    # The returned values of `stats.linregress` is:
    #   (slope, intercept, rvalue, pvalue, stderr, intercept_stderr)
    rhat = stats.linregress(np.arange(len(cum_log_returns)), cum_log_returns)[2]

    return rhat**2


def tail_ratio(returns: pd.Series | np.ndarray) -> float:
    """Determines the ratio between the right (95%) and left tail (5%).

    For example, a ratio of 0.25 means that losses are four times as bad as
    profits.

    Parameters
    ----------
    returns: pd.Series | np.ndarray
        Daily returns of the strategy, noncumulative.
         - See full explanation in :func:`~cofrics.stats.cum_returns`.

    Returns
    -------
    tail_ratio: float
    """

    if len(returns) < 1:
        return np.nan

    returns = np.asanyarray(returns)
    # Be tolerant of nan's
    returns = returns[~np.isnan(returns)]
    if len(returns) < 1:
        return np.nan

    return np.abs(np.percentile(returns, 95)) / np.abs(np.percentile(returns, 5))


def capture(returns, factor_returns, period=DAILY) -> float:
    """Compute capture ratio.

    Parameters
    ----------
    returns: pd.Series | np.ndarray
        Returns of the strategy, noncumulative.
        - See full explanation in :func:`~cofrics.stats.cum_returns`.

    factor_returns: pd.Series | np.ndarray
        Noncumulative returns of the factor to which beta is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.

    period: str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'monthly':12
            'weekly': 52
            'daily': 252

    Returns
    -------
    capture_ratio: float

    References
    ----------
    See https://www.investopedia.com/terms/u/up-market-capture-ratio.asp for
    details.
    """
    return annual_return(returns, period=period) / annual_return(
        factor_returns, period=period
    )


def beta_fragility_heuristic(
    returns: pd.Series | np.ndarray, factor_returns: pd.Series | np.ndarray
) -> float:
    """Estimate fragility to drops in beta.

    Parameters
    ----------
    returns: pd.Series | np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~cofrics.stats.cum_returns`.
    factor_returns: pd.Series | np.ndarray
         Daily noncumulative returns of the factor to which beta is
         computed. Usually a benchmark such as the market.
         - This is in the same style as returns.

    Returns
    -------
    float | np.nan
        The beta fragility of the strategy.

    References
    ----------
    A negative return value indicates potential losses could follow volatility
    in beta.
    The magnitude of the negative value indicates the size of the potential
    loss.

    See also::
    `A New Heuristic Measure of Fragility and Tail Risks: Application to
    Stress Testing`
        https://www.imf.org/external/pubs/ft/wp/2012/wp12216.pdf
    An IMF Working Paper describing the heuristic.
    """
    if len(returns) < 3 or len(factor_returns) < 3:
        return np.nan

    return beta_fragility_heuristic_aligned(*_aligned_series(returns, factor_returns))


def beta_fragility_heuristic_aligned(
    returns: pd.Series | np.ndarray, factor_returns: pd.Series | np.ndarray
) -> float:
    """Estimate fragility to drops in beta.

    Parameters
    ----------
    returns: pd.Series | np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~cofrics.stats.cum_returns`.

    factor_returns: pd.Series | np.ndarray
         Daily noncumulative returns of the factor to which beta is
         computed. Usually a benchmark such as the market.
         - This is in the same style as returns.

    Returns
    -------
    float | np.nan
        The beta fragility of the strategy.

    References
    ----------
    If they are pd.Series, expects returns and factor_returns have already
    been aligned on their labels.
    If np.ndarray, these arguments should have the same shape.

    See also::
    `A New Heuristic Measure of Fragility and Tail Risks: Application to
    Stress Testing`
        https://www.imf.org/external/pubs/ft/wp/2012/wp12216.pdf
    An IMF Working Paper describing the heuristic.
    """
    if len(returns) < 3 or len(factor_returns) < 3:
        return np.nan

    # Combine returns and factor returns into pairs
    returns_series = pd.Series(returns)
    factor_returns_series = pd.Series(factor_returns)
    pairs = pd.concat([returns_series, factor_returns_series], axis=1)
    pairs.columns = ["returns", "factor_returns"]

    # Exclude any rows where returns are nan.
    pairs.dropna(inplace=True)
    # Sort by beta
    pairs.sort_values(by="factor_returns", kind="stable", inplace=True)

    # Find the three vectors, using median of 3.
    start_index, mid_index, end_index = 0, len(pairs) // 2, len(pairs) - 1

    (start_returns, start_factor_returns) = pairs.iloc[start_index]
    (end_returns, end_factor_returns) = pairs.iloc[end_index]
    if len(pairs) % 2 == 1:
        (mid_returns, mid_factor_returns) = pairs.iloc[mid_index]
    else:
        mid_returns_1, mid_factor_returns_1 = pairs.iloc[mid_index - 1]
        mid_returns_2, mid_factor_returns_2 = pairs.iloc[mid_index]
        mid_returns = (mid_returns_1 + mid_returns_2) / 2
        mid_factor_returns = (mid_factor_returns_1 + mid_factor_returns_2) / 2

    factor_returns_range = end_factor_returns - start_factor_returns
    start_returns_weight = 0.5
    end_returns_weight = 0.5

    # Find weights for the start and end returns using a convex combination.
    if factor_returns_range != 0:
        start_returns_weight = (
            mid_factor_returns - start_factor_returns
        ) / factor_returns_range
        end_returns_weight = (
            end_factor_returns - mid_factor_returns
        ) / factor_returns_range

    # Calculate fragility heuristic
    weighted_start_returns = start_returns_weight * start_returns
    weighted_end_returns = end_returns_weight * end_returns
    heuristic = weighted_start_returns + weighted_end_returns - mid_returns

    return heuristic


def gpd_risk_estimates(returns: pd.Series | np.ndarray, var_p: float = 0.01):
    """Estimate VaR and ES using the Generalized Pareto Distribution (GPD).

    Parameters
    ----------
    returns: pd.Series | np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~cofrics.stats.cum_returns`.

    var_p: float
        The percentile to use for estimating the VaR and ES.

    Returns
    -------
    [threshold, scale_param, shape_param, var_estimate, es_estimate]
       : list[float]
        threshold - the threshold use to cut off exception tail losses.
        scale_param - a parameter (often denoted by sigma, capturing the
            scale, related to variance).
        shape_param - a parameter (often denoted by xi, capturing the shape or
            type of the distribution).
        var_estimate - an estimate for the VaR for the given percentile.
        es_estimate - an estimate for the ES for the given percentile.

    References
    ----------
    See also::
    `An Application of Extreme Value Theory for Measuring Risk
    <https://link.springer.com/article/10.1007/s10614-006-9025-7>`
    A paper describing how to use the Generalized Pareto Distribution to
    estimate VaR and ES.
    """
    if len(returns) < 3:
        result = np.zeros(5)
        if isinstance(returns, pd.Series):
            result = pd.Series(result)
        return result

    return gpd_risk_estimates_aligned(*_aligned_series(returns, var_p))


def gpd_risk_estimates_aligned(returns: pd.Series | np.ndarray, var_p: float = 0.01):
    """Estimate VaR and ES using the Generalized Pareto Distribution (GPD).

    Parameters
    ----------
    returns: pd.Series | np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~cofrics.stats.cum_returns`.
    var_p: float
        The percentile to use for estimating the VaR and ES

    Returns
    -------
    [threshold, scale_param, shape_param, var_estimate, es_estimate]: list[float]
        threshold - the threshold use to cut off exception tail losses
        scale_param - a parameter (often denoted by sigma, capturing the
            scale, related to variance)
        shape_param - a parameter (often denoted by xi, capturing the shape or
            type of the distribution)
        var_estimate - an estimate for the VaR for the given percentile
        es_estimate - an estimate for the ES for the given percentile

    References
    ----------
    See also::
    `An Application of Extreme Value Theory for
    Measuring Risk <https://link.springer.com/article/10.1007/s10614-006-9025-7>`
    A paper describing how to use the Generalized Pareto Distribution to
    estimate VaR and ES.
    """
    result = np.zeros(5)
    if len(returns) >= 3:
        try:
            returns_array = pd.Series(returns).to_numpy()
        except AttributeError:
            # Support for pandas < 0.25
            returns_array = pd.Series(returns).as_matrix()

        flipped_returns = -1 * returns_array
        losses = flipped_returns[flipped_returns > 0]
        threshold = DEFAULT_THRESHOLD
        finished = False
        scale_param = shape_param = 0
        while not finished and threshold > MINIMUM_THRESHOLD:
            losses_beyond_threshold = losses[losses >= threshold]
            param_result = gpd_log_likelihood_minimizer_aligned(losses_beyond_threshold)
            if param_result[0] and param_result[1]:
                scale_param, shape_param = param_result[0], param_result[1]
                var_estimate = gpd_var_calculator(
                    threshold,
                    scale_param,
                    shape_param,
                    var_p,
                    len(losses),
                    len(losses_beyond_threshold),
                )
                # non-negative shape parameter is required for fat tails
                # non-negative VaR estimate is required for loss of some kind
                if shape_param > 0 and var_estimate > 0:
                    finished = True
            if not finished:
                threshold = threshold / 2
        if finished:
            es_estimate = gpd_es_calculator(
                var_estimate, threshold, scale_param, shape_param
            )
            result = np.array(
                [threshold, scale_param, shape_param, var_estimate, es_estimate]
            )

    return pd.Series(result) if isinstance(returns, pd.Series) else result


def gpd_es_calculator(var_estimate, threshold, scale_param, shape_param):
    result = 0
    if shape_param != 1:
        # This formula is from Gilli and Kellezi pg.8
        var_ratio = var_estimate / (1 - shape_param)
        param_ratio = (scale_param - (shape_param * threshold)) / (1 - shape_param)
        result = var_ratio + param_ratio

    return result


def gpd_var_calculator(
    threshold, scale_param, shape_param, probability, total_n, exceedance_n
):
    result = 0
    if exceedance_n > 0 and shape_param > 0:
        # This formula is from Gilli and Kellezi pg. 12
        param_ratio = scale_param / shape_param
        prob_ratio = (total_n / exceedance_n) * probability
        result = threshold + (param_ratio * (pow(prob_ratio, -shape_param) - 1))

    return result


def gpd_log_likelihood_minimizer_aligned(price_data):
    result = [False, False]
    default_scale_param = default_shape_param = 1
    if len(price_data) > 0:
        gpd_log_likelihood_lambda = gpd_log_likelihood_factory(price_data)
        optimization_results = optimize.minimize(
            gpd_log_likelihood_lambda,
            np.array([default_scale_param, default_shape_param]),
            method="Nelder-Mead",
        )
        if optimization_results.success:
            resulting_params = optimization_results.x
            if len(resulting_params) == 2:
                result[:2] = resulting_params

    return result


def gpd_log_likelihood_factory(price_data) -> callable:
    return lambda params: gpd_log_likelihood(params, price_data)


def gpd_log_likelihood(params, price_data):
    if params[1] != 0:
        return -gpd_log_likelihood_scale_and_shape(params[0], params[1], price_data)
    else:
        return -gpd_log_likelihood_scale_only(params[0], price_data)


def gpd_log_likelihood_scale_and_shape_factory(price_data) -> callable:
    # Minimize a function of two variables requires a list of params.
    # We are expecting the lambda below to be called as follows:
    #   parameters = [scale, shape]
    # The final outer negative is added because scipy only minimizes.
    return lambda params: -gpd_log_likelihood_scale_and_shape(
        params[0], params[1], price_data
    )


def gpd_log_likelihood_scale_and_shape(scale, shape, price_data):
    n = len(price_data)
    result = -1 * float_info.max
    if scale != 0:
        param_factor = shape / scale
        if shape != 0 and param_factor >= 0 and scale >= 0:
            result = -n * np.log(scale) - (
                ((1 / shape) + 1) * (np.log((shape / scale * price_data) + 1)).sum()
            )
    return result


def gpd_log_likelihood_scale_only_factory(price_data) -> callable:
    # The negative is added because scipy only minimizes.
    return lambda scale: -gpd_log_likelihood_scale_only(scale, price_data)


def gpd_log_likelihood_scale_only(scale, price_data):
    n = len(price_data)
    data_sum = price_data.sum()
    result = -1 * float_info.max
    if scale >= 0:
        result = -n * np.log(scale) - data_sum / scale
    return result


def up_capture(
    returns: pd.Series | np.ndarray, factor_returns: pd.Series | np.ndarray, **kwargs
) -> float:
    """Compute the capture ratio for periods when the benchmark return is
    positive.

    Parameters
    ----------
    returns: pd.Series | np.ndarray
        Returns of the strategy, noncumulative.
        - See full explanation in :func:`~cofrics.stats.cum_returns`.

    factor_returns: pd.Series | np.ndarray
        Noncumulative returns of the factor to which beta is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.

    period (keyword): str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'monthly':12
            'weekly': 52
            'daily': 252

    Returns
    -------
    up_capture: float

    References
    ----------
    See https://www.investopedia.com/terms/u/up-market-capture-ratio.asp for
    more information.
    """
    return up(returns, factor_returns, function=capture, **kwargs)


def down_capture(
    returns: pd.Series | np.ndarray, factor_returns: pd.Series | np.ndarray, **kwargs
) -> float:
    """Compute the capture ratio for periods when the benchmark return is
    negative.

    Parameters
    ----------
    returns: pd.Series | np.ndarray
        Returns of the strategy, noncumulative.
        - See full explanation in :func:`~cofrics.stats.cum_returns`.

    factor_returns: pd.Series | np.ndarray
        Noncumulative returns of the factor to which beta is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.

    period (keyword): str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'monthly':12
            'weekly': 52
            'daily': 252

    Returns
    -------
    down_capture: float

    References
    ----------
    See https://www.investopedia.com/terms/d/down-market-capture-ratio.asp for
    more information.
    """
    return down(returns, factor_returns, function=capture, **kwargs)


def up_down_capture(
    returns: pd.Series | np.ndarray, factor_returns: pd.Series | np.ndarray, **kwargs
) -> float:
    """Computes the ratio of up_capture to down_capture.

    Parameters
    ----------
    returns: pd.Series | np.ndarray
        Returns of the strategy, noncumulative.
        - See full explanation in :func:`~cofrics.stats.cum_returns`.

    factor_returns: pd.Series | np.ndarray
        Noncumulative returns of the factor to which beta is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.

    period (keyword): str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'monthly':12
            'weekly': 52
            'daily': 252

    Returns
    -------
    up_down_capture: float
        the up down capture ratio
    """
    return up_capture(returns, factor_returns, **kwargs) / down_capture(
        returns, factor_returns, **kwargs
    )


def up_alpha_beta(
    returns: pd.Series | np.ndarray, factor_returns: pd.Series | np.ndarray, **kwargs
) -> tuple[float, float]:
    """Computes alpha and beta for periods when the benchmark return is
    positive.

    Parameters
    ----------
    see documentation for `alpha_beta`.

    Returns
    -------
    float
        Alpha.
    float
        Beta.
    """
    return up(returns, factor_returns, function=alpha_beta_aligned, **kwargs)


def down_alpha_beta(
    returns: pd.Series | np.ndarray, factor_returns: pd.Series | np.ndarray, **kwargs
) -> tuple[float, float]:
    """Computes alpha and beta for periods when the benchmark return is
    negative.

    Parameters
    ----------
    see documentation for `alpha_beta`.

    Returns
    -------
    alpha: float
    beta: float
    """
    return down(returns, factor_returns, function=alpha_beta_aligned, **kwargs)


def roll_up_capture(
    returns, factor_returns, window=10, **kwargs
) -> pd.Series | np.ndarray:
    """Computes the up capture measure over a rolling window.

    See documentation for :func:`~cofrics.stats.up_capture`.
    (pass all args, kwargs required)

    Parameters
    ----------
    returns: pd.Series | np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~cofrics.stats.cum_returns`.

    factor_returns: pd.Series | np.ndarray
        Noncumulative returns of the factor to which beta is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.

    window: int, required
        Size of the rolling window in terms of the periodicity of the data.
        - e.g. window = 60, periodicity=DAILY, represents a rolling 60-day
        window.
    """
    return roll(returns, factor_returns, window=window, function=up_capture, **kwargs)


def roll_down_capture(
    returns, factor_returns, window=10, **kwargs
) -> pd.Series | np.ndarray:
    """Computes the down capture measure over a rolling window.

    see documentation for :func:`~cofrics.stats.down_capture`.
    (pass all args, kwargs required)

    Parameters
    ----------
    returns: pd.Series | np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~cofrics.stats.cum_returns`.

    factor_returns: pd.Series | np.ndarray
        Noncumulative returns of the factor to which beta is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.

    window: int, required
        Size of the rolling window in terms of the periodicity of the data.
        - e.g. window = 60, periodicity=DAILY, represents a rolling 60-day
        window.
    """
    return roll(returns, factor_returns, window=window, function=down_capture, **kwargs)


def roll_up_down_capture(
    returns, factor_returns, window=10, **kwargs
) -> pd.Series | np.ndarray:
    """Computes the up/down capture measure over a rolling window.

    Parameters
    ----------
    returns: pd.Series | np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~cofrics.stats.cum_returns`.

    factor_returns: pd.Series | np.ndarray
        Noncumulative returns of the factor to which beta is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.

    window: int, required
        Size of the rolling window in terms of the periodicity of the data.
        - e.g. window = 60, periodicity=DAILY, represents a rolling 60-day
        window.

    See Also
    --------
    documentation for :func:`~cofrics.stats.up_down_capture`.
    (pass all args, kwargs required)
    """
    return roll(
        returns, factor_returns, window=window, function=up_down_capture, **kwargs
    )


def value_at_risk(returns, cutoff: float = 0.05) -> float:
    """Value at risk (VaR) of a returns stream.

    Parameters
    ----------
    returns: pandas.Series | 1-D np.array
        Non-cumulative daily returns.

    cutoff: float, optional
        Decimal representing the percentage cutoff for the bottom percentile
        of returns. Defaults to 0.05.

    Returns
    -------
    VaR: float
        The VaR value.
    """
    return np.percentile(returns, 100.0 * cutoff)


def conditional_value_at_risk(returns: pd.Series | np.ndarray, cutoff: float = 0.05):
    """Conditional value at risk (CVaR) of a returns stream.

    CVaR measures the expected single-day returns of an asset on that asset's
    worst performing days, where "worst-performing" is defined as falling below
    ``cutoff`` as a percentile of all daily returns.

    Parameters
    ----------
    returns: pandas.Series | 1-D np.array
        Non-cumulative daily returns.

    cutoff: float, optional
        Decimal representing the percentage cutoff for the bottom percentile
        of returns. Defaults to 0.05.

    Returns
    -------
    CVaR: float
        The CVaR value.
    """
    # Note: [PERF] Instead of using the 'value_at_risk' function to find the
    #  cutoff value, which requires a call to numpy.percentile, determine the
    #  cutoff index manually and partition out the lowest returns values.
    #
    #  The value at the cutoff index should be included in the partition.
    cutoff_index = int((len(returns) - 1) * cutoff)

    return np.mean(np.partition(returns, cutoff_index)[: cutoff_index + 1])


SIMPLE_STAT_FUNCS = [
    cum_returns_final,
    annual_return,
    annual_volatility,
    sharpe_ratio,
    calmar_ratio,
    stability_of_timeseries,
    max_drawdown,
    omega_ratio,
    sortino_ratio,
    stats.skew,
    stats.kurtosis,
    tail_ratio,
    cagr,
    value_at_risk,
    conditional_value_at_risk,
]

FACTOR_STAT_FUNCS = [
    excess_sharpe,
    alpha,
    beta,
    beta_fragility_heuristic,
    gpd_risk_estimates,
    capture,
    up_capture,
    down_capture,
]
