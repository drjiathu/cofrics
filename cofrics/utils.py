from __future__ import annotations

import re
import warnings
# from datetime import datetime
from functools import wraps
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from IPython.display import display
from numpy.lib.stride_tricks import as_strided
from pandas.tseries.offsets import BDay, CDay, Day
from scipy.stats import mode

# from .deprecate import deprecated

try:
    # Faster versions
    import bottleneck as bn

    def _wrap_function(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            out = kwargs.pop("out", None)
            data = f(*args, **kwargs)
            if out is None:
                out = data
            else:
                out[()] = data

            return out

        return wrapped

    nanmean = _wrap_function(bn.nanmean)
    nanstd = _wrap_function(bn.nanstd)
    nansum = _wrap_function(bn.nansum)
    nanmax = _wrap_function(bn.nanmax)
    nanmin = _wrap_function(bn.nanmin)
    nanargmax = _wrap_function(bn.nanargmax)
    nanargmin = _wrap_function(bn.nanargmin)
except ImportError:
    # Slower numpy version
    nanmean = np.nanmean
    nanstd = np.nanstd
    nansum = np.nansum
    nanmax = np.nanmax
    nanmin = np.nanmin
    nanargmax = np.nanargmax
    nanargmin = np.nanargmin

DAYS_OF_THE_WEEK = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


class NotMatchingTimezoneError(Exception):
    pass


class MaxLossExceededError(Exception):
    pass


def rethrow(exception, additional_message):
    """Re-raise the last exception that was active in the current scope
    without losing the stacktrace but adding additional message.

    This is hacky because it has to be compatible with both python 2/3.
    """

    e, m = exception, additional_message
    if not e.args:
        e.args = (m,)
    else:
        e.args = (e.args[0] + m,) + e.args[1]

    raise e


def non_unique_bin_edges_error(func):
    """Give user a more informative error in case it is not possible to
    properly calculate quantiles on the input DataFrame (factor).
    """

    message = """

    An error occurred while computing bins/quantiles on the input provided.
    This usually happens when the input contains too many identical values and
    they span more than one quantile.

    The quantiles are chosen to have the same number of records each, but the
    same value cannot span multiple quantiles. Possible workarounds are:

    1 - Decrease the number of quantiles
    2 - Specify a custom quantiles range, e.g. [0, .50, .75, 1.] to get
        unequal number of records per quantile
    3 - Use 'bins' option instead of 'quantiles'. 'bins' chooses the buckets
        to be evenly spaced according to the VALUES themselves, while
        'quantiles' forces the buckets to have the same number of records.
    4 - For factors with discrete values use the 'bins' option with custom
        ranges and create a range for each discrete value.

    Please see utils.get_clean_factor_and_forward_returns documentation for
    full documentation of 'bins' and 'quantiles' options.

    """

    def dec(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            if "Bin edges must be unique" in str(e):
                rethrow(e, message)
            raise

    return dec


@non_unique_bin_edges_error
def quantize_factor(
    factor_data: pd.DataFrame,
    quantiles: int | Sequence[float] = 5,
    bins=None,
    by_group=False,
    no_raise=False,
    zero_aware=False,
) -> pd.Series:
    """Computes period wise factor quantiles.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.

        - See full explanation in utils.get_clean_factor_and_forward_returns

    quantiles : int or Sequence[float]
        Number of equal-sized quantile buckets to use in factor bucketing.
        Alternately sequence of quantiles, allowing non-equal-sized buckets
        e.g. [0, .10, .5, .90, 1.] or [.05, .5, .95]
        Only one of 'quantiles' or 'bins' can be not-None
    bins : int or Sequence[float]
        Number of equal-width (value-wise) bins to use in factor bucketing.
        Alternately sequence of bin edges allowing for non-uniform bin width
        e.g. [-4, -2, -0.5, 0, 10]
        Only one of 'quantiles' or 'bins' can be not-None
    by_group : bool, by default False.
        If True, compute quantile buckets separately for each group.
    no_raise: bool, by default False.
        If True, no exceptions are thrown and the values for which the
        exception would have been thrown are set to np.NaN
    zero_aware : bool, by default False.
        If True, compute quantile buckets separately for positive and negative
        signal values.
        Note: This is useful if your signal is centered and zero is the
            separation between long and short signals, respectively.

    Returns
    -------
    factor_quantile : pd.Series
        Factor quantiles indexed by date and asset.
    """

    if not (
        (quantiles is not None and bins is None)
        or (quantiles is None and bins is not None)
    ):
        raise ValueError("Either quantiles or bins should be provided")

    if zero_aware and not (isinstance(quantiles, int) or isinstance(bins, int)):
        msg = "zero_aware should only be True when quantiles or bins is an integer"
        raise ValueError(msg)

    def quantile_calc(x, _quantiles, _bins, _zero_aware, _no_raise):
        try:
            if _quantiles is not None and _bins is None and not _zero_aware:
                return pd.qcut(x, _quantiles, labels=False) + 1
            elif _quantiles is not None and _bins is None and _zero_aware:
                pos_quantiles = (
                    pd.qcut(x[x >= 0], _quantiles // 2, labels=False)
                    + _quantiles // 2
                    + 1
                )
                neg_quantiles = pd.qcut(x[x < 0], _quantiles // 2, labels=False) + 1
                return pd.concat([pos_quantiles, neg_quantiles]).sort_index()
            elif _bins is not None and _quantiles is None and not _zero_aware:
                return pd.cut(x, _bins, labels=False) + 1
            elif _bins is not None and _quantiles is None and _zero_aware:
                pos_bins = pd.cut(x[x >= 0], _bins // 2, labels=False) + _bins // 2 + 1
                neg_bins = pd.cut(x[x < 0], _bins // 2, labels=False) + 1
                return pd.concat([pos_bins, neg_bins]).sort_index()
        except Exception as e:
            if _no_raise:
                return pd.Series(index=x.index)
            raise e

    grouper = [factor_data.index.get_level_values("date")]

    if by_group:
        grouper.append("group")

    factor_quantile = factor_data.groupby(grouper, group_keys=False, observed=False)[
        "factor"
    ].apply(quantile_calc, quantiles, bins, zero_aware, no_raise)
    factor_quantile.name = "factor_quantile"

    return factor_quantile.dropna()


def infer_trading_calender(
    factor_idx: pd.DatetimeIndex, prices_idx: pd.DatetimeIndex
) -> CDay:
    """Infer the trading calendar from factor and price information.

    Parameters
    ----------
    factor_idx : pd.DatetimeIndex
        The factor datetimes for which we are computing the forward returns
    prices_idx : pd.DatetimeIndex
        The prices datetimes associated with the factor data

    Returns
    -------
    # TODO: make sure of return_type.
    calendar : pd.tseries.CDay
    """

    full_idx = factor_idx.union(prices_idx)

    traded_weekdays, holidays = [], []

    for day, day_str in enumerate(DAYS_OF_THE_WEEK):
        weekday_mask = full_idx.dayofweek == day

        # Drop days of the week that are not traded at all
        if not weekday_mask.any():
            continue
        traded_weekdays.append(day_str)

        # Look for holidays
        used_weekdays = full_idx[weekday_mask].normalize()
        all_weekdays = pd.date_range(
            start=full_idx.min(), end=full_idx.max(), freq=CDay(weekmask=day_str)
        ).normalize()
        _holidays = all_weekdays.difference(used_weekdays)
        _holidays = [timestamp.date() for timestamp in _holidays]
        holidays.extend(_holidays)

    traded_weekdays = " ".join(traded_weekdays)

    return CDay(weekmask=traded_weekdays, holidays=holidays)


def compute_forward_returns(
    factor: pd.Series,
    prices: pd.DataFrame,
    periods: Sequence[int] = (1, 5, 10),
    filter_zscore: int | float | None = None,
    cumulative_returns: bool = True,
) -> pd.DataFrame:
    """Finds the N period forward returns (as percentage change) for each
    asset provided.

    Parameters
    ----------
    factor : pd.Series - MultiIndex
        A MultiIndex Series indexed by timestamp (level 0) and asset (level 1),
        containing the values for a single alpha factor.

        - See full explanation in utils.get_clean_factor_and_forward_returns

    prices : pd.DataFrame
        Pricing data to use in forward price calculation.
        Assets as columns, dates as index. Pricing data must span the factor
        analysis time period plus an additional buffer window that is greater
        than the maximum number of expected periods in the forward returns
        calculations.
    periods : Sequence[int]
        periods to compute forward returns on.
    filter_zscore : int or float, optional
        Sets forward returns greater than X standard deviations
        from the mean to nan. Set it to 'None' to avoid filtering.
        Caution: this outlier filtering incorporates lookahead bias.
    cumulative_returns : bool, by default True
        If True, forward returns columns will contain cumulative returns.
        Setting this to False is useful if you want to analyze how predictive
        a factor is for a single forward day.

    Returns
    -------
    forward_returns : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by timestamp (level 0) and asset
        (level 1), containing the forward returns for assets.
        Forward returns column names follow the format accepted by
        pd.Timedelta (e.g. '1D', '30m', '3h15m', '1D1h', etc.).
        'date' index freq property (forward_returns.index.levels[0].freq)
        will be set to a trading calendar (pandas DateOffset) inferred
        from the input data (see infer_trading_calendar for more details).
    """

    factor_dateindex = factor.index.levels[0]

    if factor_dateindex.tz != prices.index.tz:
        raise NotMatchingTimezoneError(
            "The timezone of 'factor' is not the same as the timezone of "
            "'prices'. See the pandas methods tz_localize and tz_convert."
        )

    freq = infer_trading_calender(factor_dateindex, prices.index)
    factor_dateindex = factor_dateindex.intersection(prices.index)

    if len(factor_dateindex) == 0:
        raise ValueError(
            "Factor and prices indices don't match: make sure they have the "
            "same convention in terms of datetimes and symbol-names"
        )

    # Chop prices down to only the assets we care about
    # (= unique assets in `factor`).
    # We could modify `prices` in place, but that might confuse the caller.
    prices = prices.filter(items=factor.index.levels[1])

    raw_values_dict, column_list = {}, []

    for period in sorted(periods):
        if cumulative_returns:
            returns = prices.pct_change(period)
        else:
            returns = prices.pct_change()

        forward_returns = returns.shift(-period).reindex(factor_dateindex)

        if filter_zscore is not None:
            mask = (
                abs(forward_returns - forward_returns.mean())
                > filter_zscore * forward_returns.std()
            )
            forward_returns[mask] = np.nan

        # Find the period length, which will be the column name.
        # We'll test several entries in order to find out the most likely
        # period length (in case the user passed inconsistent data).
        days_diffs = []
        for i in range(30):
            if i >= len(forward_returns.index):
                break
            p_idx = prices.index.get_loc(forward_returns.index[i])
            if p_idx is None or p_idx < 0 or (p_idx + period) >= len(prices.index):
                continue
            start = prices.index[p_idx]
            end = prices.index[p_idx + period]
            period_len = diff_custom_calendar_timedeltas(start, end, freq)
            days_diffs.append(period_len.days)

        delta_days = period_len.days - mode(days_diffs, keepdims=False).mode
        period_len -= pd.Timedelta(days=delta_days)
        label = timedelta_to_string(period_len)

        column_list.append(label)

        raw_values_dict[label] = np.concatenate(forward_returns.values)

    df = pd.DataFrame.from_dict(raw_values_dict)
    df.set_index(
        pd.MultiIndex.from_product(
            [factor_dateindex, prices.columns], names=["date", "asset"]
        ),
        inplace=True,
    )
    df = df.reindex(factor.index)

    # Now set the columns correctly.
    df = df[column_list]
    df.index.levels[0].freq = freq
    df.index.set_names(["date", "asset"], inplace=True)

    return df


def backshift_returns_series(series: pd.Series, n: int):
    """Shift a multi-indexed series backwards by n observations in the first
    level.

    This can be used to convert backward-looking returns into a
    forward-returns series.

    Parameters
    ----------
    series: pd.Series

    n: int

    Returns
    -------

    """
    idx = series.index
    dates, sids = idx.levels
    date_labels, sid_labels = map(np.array, idx.labels)

    # Output date labels will contain all but the last N dates.
    new_dates = dates[:-n]

    # Output data will remove the first M rows, where M is the index of the
    # last record with one of the first N dates.
    cutoff = date_labels.searchsorted(n)
    new_date_labels = date_labels[cutoff:] - n
    new_sid_labels = sid_labels[cutoff:]
    new_values = series.values[cutoff:]

    assert new_date_labels[0] == 0

    new_index = pd.MultiIndex(
        levels=[new_dates, sids],
        # TODO: labels or codes
        codes=[new_date_labels, new_sid_labels],
        sortorder=1,
        names=idx.names,
    )

    return pd.Series(data=new_values, index=new_index)


def demean_forward_returns(factor_data: pd.DataFrame, grouper=None):
    """Convert forward returns to returns relative to mean period wise
    all-universe or group returns.

    Group-wise normalization incorporates the assumption of a group neutral
    portfolio constraint and thus allows the factor to be evaluated across
    groups.

    For example, if AAPL 5 period return is 0.1% and mean 5 period return for
    the Technology stocks in our universe was 0.5% in the same period, the
    group adjusted 5 period return for AAPL in this period is -0.4%.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        Forward returns indexed by date and asset.
        Separate column for each forward return window.
    grouper : list
        If True, demean according to group.

    Returns
    -------
    adjusted_forward_returns : pd.DataFrame - MultiIndex
        DataFrame of the same format as the input, but with each
        security's returns normalized by group.
    """

    factor_data = factor_data.copy()

    if not grouper:
        grouper = factor_data.index.get_level_values("date")

    cols = get_forward_returns_columns(factor_data.columns)
    factor_data[cols] = factor_data.groupby(grouper, observed=False)[cols].transform(
        lambda x: x - x.mean()
    )

    return factor_data


def print_table(table: pd.DataFrame | pd.Series, name: str = None, fmt: str = None):
    """Pretty print a pd.DataFrame.

    Use HTML output if running inside Jupyter Notebook, otherwise formatted
    test output.

    Parameters
    ----------
    table : pd.Series or pd.DataFrame
        Table to pretty-print.
    name : str, optional
        Table name to display in upper left corner.
    fmt : str, optional
        Formatter to use for displaying table elements.
        E.g. '{0:.2f}%' for displaying 100 as '100.00%'.
        Restores original setting after displaying.

    Returns
    -------

    """
    if isinstance(table, pd.Series):
        table = pd.DataFrame(table)

    if isinstance(table, pd.DataFrame):
        table.columns.name = name

    prev_option = pd.get_option("display.float_format")

    if fmt is not None:
        pd.set_option("display.float_format", lambda x: fmt.format(x))

    display(table)

    if fmt is not None:
        pd.set_option("display.float_format", prev_option)


def get_clean_factor(
    factor: pd.Series,
    forward_returns: pd.DataFrame,
    groupby: pd.Series = None,
    binning_by_group=False,
    quantiles=5,
    bins=None,
    groupby_labels=None,
    max_loss=0.35,
    zero_aware=False,
) -> pd.DataFrame:
    """Formats the factor data, forward return data, and group mappings into a
    DataFrame that contains aligned MultiIndex indices of timestamp and asset.

    The returned data will be formatted to be suitable for Caliper functions.

    It is safe to skip a call to this function and still make use of Caliper
    functionalities as long as the factor data conforms to the format returned
    from get_clean_factor_and_forward_returns and documented here

    Parameters
    ----------
    factor : pd.Series - MultiIndex
        A MultiIndex Series indexed by timestamp (level 0) and asset
        (level 1), containing the values for a single alpha factor
        ::
            -----------------------------------
                date    |    asset   |
            -----------------------------------
                        |   AAPL     |   0.5
                        -----------------------
                        |   BA       |  -1.1
                        -----------------------
            2014-01-01  |   CMG      |   1.7
                        -----------------------
                        |   DAL      |  -0.1
                        -----------------------
                        |   LULU     |   2.7
                        -----------------------

    forward_returns : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by timestamp (level 0) and asset
        (level 1), containing the forward returns for assets.
        Forward returns column names must follow the format accepted by
        pd.Timedelta (e.g. '1D', '30m', '3h15m', '1D1h', etc.).
        'date' index freq property must be set to a trading calendar
        (pandas DateOffset), see infer_trading_calendar for more details.
        This information is currently used only in cumulative returns
        computation
        ::
            ---------------------------------------
                       |       | 1D  | 5D  | 10D
            ---------------------------------------
                date   | asset |     |     |
            ---------------------------------------
                       | AAPL  | 0.09|-0.01|-0.079
                       ----------------------------
                       | BA    | 0.02| 0.06| 0.020
                       ----------------------------
            2014-01-01 | CMG   | 0.03| 0.09| 0.036
                       ----------------------------
                       | DAL   |-0.02|-0.06|-0.029
                       ----------------------------
                       | LULU  |-0.03| 0.05|-0.009
                       ----------------------------

    groupby : pd.Series - MultiIndex or dict
        Either A MultiIndex Series indexed by date and asset,
        containing the period wise group codes for each asset, or
        a dict of asset to group mappings. If a dict is passed,
        it is assumed that group mappings are unchanged for the
        entire time period of the passed factor data.
    binning_by_group : bool, by default False
        If True, compute quantile buckets separately for each group.
        This is useful when the factor values range vary considerably
        across groups so that it is wise to make the binning group relative.
        You should probably enable this if the factor is intended
        to be analyzed for a group neutral portfolio
    quantiles : int or Sequence[float], by default 5.
        Number of equal-sized quantile buckets to use in factor bucketing.
        Alternately sequence of quantiles, allowing non-equal-sized buckets
        e.g. [0, .10, .5, .90, 1.] or [.05, .5, .95]
        Only one of 'quantiles' or 'bins' can be not-None
    bins : int or Sequence[float], optional
        Number of equal-width (value-wise) bins to use in factor bucketing.
        Alternately sequence of bin edges allowing for non-uniform bin width
        e.g. [-4, -2, -0.5, 0, 10]
        Chooses the buckets to be evenly spaced according to the values
        themselves. Useful when the factor contains discrete values.
        Only one of 'quantiles' or 'bins' can be not-None
    groupby_labels : dict, optional
        A dictionary keyed by group code with values corresponding
        to the display name for each group.
    max_loss : float, by default 0.35
        Maximum percentage (0.00 to 1.00) of factor data dropping allowed,
        computed comparing the number of items in the input factor index and
        the number of items in the output DataFrame index.
        Factor data can be partially dropped due to being flawed itself
        (e.g. NaNs), not having provided enough price data to compute
        forward returns for all factor values, or because it is not possible
        to perform binning.
        Set max_loss=0 to avoid Exceptions suppression.
    zero_aware : bool, by default False
        If True, compute quantile buckets separately for positive and negative
        signal values. This is useful if your signal is centered and zero is
        the separation between long and short signals, respectively.
        'quantiles' is None.

    Returns
    -------
    merged_data : pd.DataFrame - MultiIndex
        A MultiIndex Series indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.

        - forward returns column names follow the format accepted by
          pd.Timedelta (e.g. '1D', '30m', '3h15m', '1D1h', etc.)

        - 'date' index freq property (merged_data.index.levels[0].freq) is the
          same as that of the input forward returns data. This is currently
          used only in cumulative returns computation
        ::
           -------------------------------------------------------------------
                      |       | 1D  | 5D  | 10D  |factor|group|factor_quantile
           -------------------------------------------------------------------
               date   | asset |     |     |      |      |     |
           -------------------------------------------------------------------
                      | AAPL  | 0.09|-0.01|-0.079|  0.5 |  G1 |      3
                      --------------------------------------------------------
                      | BA    | 0.02| 0.06| 0.020| -1.1 |  G2 |      5
                      --------------------------------------------------------
           2014-01-01 | CMG   | 0.03| 0.09| 0.036|  1.7 |  G2 |      1
                      --------------------------------------------------------
                      | DAL   |-0.02|-0.06|-0.029| -0.1 |  G3 |      5
                      --------------------------------------------------------
                      | LULU  |-0.03| 0.05|-0.009|  2.7 |  G1 |      2
                      --------------------------------------------------------
    """

    initial_amount = float(len(factor.index))

    factor_copy = factor.copy()
    factor_copy.index = factor_copy.index.rename(["date", "asset"])
    factor_copy = factor_copy[np.isfinite(factor_copy)]

    merged_data = forward_returns.copy()
    merged_data["factor"] = factor_copy

    if groupby is not None:
        if isinstance(groupby, dict):
            diff = set(factor_copy.index.get_level_values("asset")) - set(
                groupby.keys()
            )
            if len(diff) > 0:
                raise KeyError(f"Assets {list(diff)} not in group mapping")

            ss = pd.Series(groupby)
            groupby = pd.Series(
                index=factor_copy.index,
                data=ss[factor_copy.index.get_level_values("asset")].values,
            )

        if groupby_labels is not None:
            diff = set(groupby.values) - set(groupby_labels.keys())
            if len(diff) > 0:
                raise KeyError("groups {} not in passed group names".format(list(diff)))

            sn = pd.Series(groupby_labels)
            groupby = pd.Series(index=groupby.index, data=sn[groupby.values].values)

        merged_data["group"] = groupby.astype("category")

    merged_data = merged_data.dropna()

    fwd_ret_amount = float(len(merged_data.index))

    no_raise = False if max_loss == 0 else True
    quantile_data = quantize_factor(
        factor_data=merged_data,
        quantiles=quantiles,
        bins=bins,
        by_group=binning_by_group,
        no_raise=no_raise,
        zero_aware=zero_aware,
    )

    merged_data["factor_quantile"] = quantile_data

    merged_data = merged_data.dropna()

    binning_amount = float(len(merged_data.index))

    tot_loss = (initial_amount - binning_amount) / initial_amount
    fwd_ret_loss = (initial_amount - fwd_ret_amount) / initial_amount
    bin_loss = tot_loss - fwd_ret_loss

    print(
        "Dropped %.1f%% entries from factor data: %.1f%% in forward "
        "returns computation and %.1f%% in binning phase "
        "(set max_loss=0 to see potentially suppressed Exceptions)."
        % (tot_loss * 100, fwd_ret_loss * 100, bin_loss * 100)
    )

    if tot_loss > max_loss:
        message = "max_loss (%.1f%%) exceeded %.1f%%, consider increasing it." % (
            max_loss * 100,
            tot_loss * 100,
        )
        raise MaxLossExceededError(message)
    else:
        print("max_loss is %.1f%%, not exceeded: OK!" % (max_loss * 100))

    return merged_data


def get_clean_factor_and_forward_returns(
    factor: pd.Series,
    prices: pd.DataFrame,
    groupby: pd.Series = None,
    binning_by_group=False,
    quantiles: int | Sequence[float] = 5,
    bins: int | Sequence[float] = None,
    periods: Sequence[int] = (1, 5, 10),
    filter_zscore: int | float = 20,
    groupby_labels: dict = None,
    max_loss: float = 0.35,
    zero_aware=False,
    cumulative_returns=True,
) -> pd.DataFrame:
    """Formats the factor data, pricing data, and group mappings into a
    pd.DataFrame that contains aligned MultiIndex indices of timestamp and
    asset.

    The returned data will be formatted to be suitable for Caliper
    functions.

    It is safe to skip a call to this function and still make use of Caliper
    functionalities as long as the factor data conforms to the format returned
    from get_clean_factor_and_forward_returns and documented here.

    Parameters
    ----------
    factor : pd.Series - MultiIndex
        A MultiIndex Series indexed by timestamp (level 0) and asset
        (level 1), containing the values for a single alpha factor
        ::
            -----------------------------------
                date    |    asset   |
            -----------------------------------
                        |   AAPL     |   0.5
                        -----------------------
                        |   BA       |  -1.1
                        -----------------------
            2014-01-01  |   CMG      |   1.7
                        -----------------------
                        |   DAL      |  -0.1
                        -----------------------
                        |   LULU     |   2.7
                        -----------------------

    prices : pd.DataFrame
        A wide form Pandas DataFrame indexed by timestamp with assets
        in the columns.
        Pricing data must span the factor analysis time period plus an
        additional buffer window that is greater than the maximum number
        of expected periods in the forward returns calculations.
        It is important to pass the correct pricing data in depending on
        what time of period your signal was generated so to avoid lookahead
        bias, or  delayed calculations.
        'Prices' must contain at least an entry for each timestamp/asset
        combination in 'factor'. This entry should reflect the buy price
        for the assets ,and usually it is the next available price after the
        factor is computed ,but it can also be a later price if the factor is
        meant to be traded later (e.g. if the factor is computed at market
        open but traded 1 hour after market open the price information should
        be 1 hour after market open).
        'Prices' must also contain entries for timestamps following each
        timestamp/asset combination in 'factor', as many more timestamps
        as the maximum value in 'periods'. The asset price after 'period'
        timestamps will be considered the sell price for that asset when
        computing 'period' forward returns
        ::
            ----------------------------------------------------
                        | AAPL |  BA  |  CMG  |  DAL  |  LULU  |
            ----------------------------------------------------
               Date     |      |      |       |       |        |
            ----------------------------------------------------
            2014-01-01  |605.12| 24.58|  11.72| 54.43 |  37.14 |
            ----------------------------------------------------
            2014-01-02  |604.35| 22.23|  12.21| 52.78 |  33.63 |
            ----------------------------------------------------
            2014-01-03  |607.94| 21.68|  14.36| 53.94 |  29.37 |
            ----------------------------------------------------

    groupby : pd.Series - MultiIndex or dict
        Either A MultiIndex Series indexed by date and asset,
        containing the period wise group codes for each asset, or
        a dict of asset to group mappings. If a dict is passed,
        it is assumed that group mappings are unchanged for the
        entire time period of the passed factor data.
    binning_by_group : bool
        If True, compute quantile buckets separately for each group.
        This is useful when the factor values range vary considerably
        across groups so that it is wise to make the binning group relative.
        You should probably enable this if the factor is intended
        to be analyzed for a group neutral portfolio
    quantiles : int or sequence[float]
        Number of equal-sized quantile buckets to use in factor bucketing.
        Alternately sequence of quantiles, allowing non-equal-sized buckets
        e.g. [0, .10, .5, .90, 1.] or [.05, .5, .95]
        Only one of 'quantiles' or 'bins' can be not-None
    bins : int or sequence[float]
        Number of equal-width (value-wise) bins to use in factor bucketing.
        Alternately sequence of bin edges allowing for non-uniform bin width
        e.g. [-4, -2, -0.5, 0, 10]
        Chooses the buckets to be evenly spaced according to the values
        themselves. Useful when the factor contains discrete values.
        Only one of 'quantiles' or 'bins' can be not-None
    periods : sequence[int]
        periods to compute forward returns on.
    filter_zscore : int or float, optional
        Sets forward returns greater than X standard deviations
        from the mean to nan. Set it to 'None' to avoid filtering.
        Caution: this outlier filtering incorporates lookahead bias.
    groupby_labels : dict
        A dictionary keyed by group code with values corresponding
        to the display name for each group.
    max_loss : float, optional
        Maximum percentage (0.00 to 1.00) of factor data dropping allowed,
        computed comparing the number of items in the input factor index and
        the number of items in the output DataFrame index.
        Factor data can be partially dropped due to being flawed itself
        (e.g. NaNs), not having provided enough price data to compute
        forward returns for all factor values, or because it is not possible
        to perform binning.
        Set max_loss=0 to avoid Exceptions suppression.
    zero_aware : bool, optional
        If True, compute quantile buckets separately for positive and negative
        signal values. This is useful if your signal is centered and zero is
        the separation between long and short signals, respectively.
    cumulative_returns : bool, optional
        If True, forward returns columns will contain cumulative returns.
        Setting this to False is useful if you want to analyze how predictive
        a factor is for a single forward day.

    Returns
    -------
    merged_data : pd.DataFrame - MultiIndex
        A MultiIndex Series indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - forward returns column names follow  the format accepted by
          pd.Timedelta (e.g. '1D', '30m', '3h15m', '1D1h', etc.)
        - 'date' index freq property (merged_data.index.levels[0].freq) will be
          set to a trading calendar (pandas DateOffset) inferred from the input
          data (see infer_trading_calendar for more details). This is currently
          used only in cumulative returns computation
        ::
           -------------------------------------------------------------------
                      |       | 1D  | 5D  | 10D  |factor|group|factor_quantile
           -------------------------------------------------------------------
               date   | asset |     |     |      |      |     |
           -------------------------------------------------------------------
                      | AAPL  | 0.09|-0.01|-0.079|  0.5 |  G1 |      3
                      --------------------------------------------------------
                      | BA    | 0.02| 0.06| 0.020| -1.1 |  G2 |      5
                      --------------------------------------------------------
           2014-01-01 | CMG   | 0.03| 0.09| 0.036|  1.7 |  G2 |      1
                      --------------------------------------------------------
                      | DAL   |-0.02|-0.06|-0.029| -0.1 |  G3 |      5
                      --------------------------------------------------------
                      | LULU  |-0.03| 0.05|-0.009|  2.7 |  G1 |      2
                      --------------------------------------------------------

    See Also
    --------
    utils.get_clean_factor
        For use when forward returns are already available.
    """
    forward_returns = compute_forward_returns(
        factor,
        prices,
        periods,
        filter_zscore,
        cumulative_returns,
    )

    factor_data = get_clean_factor(
        factor,
        forward_returns,
        groupby=groupby,
        groupby_labels=groupby_labels,
        quantiles=quantiles,
        bins=bins,
        binning_by_group=binning_by_group,
        max_loss=max_loss,
        zero_aware=zero_aware,
    )

    return factor_data


def rate_of_return(period_ret: pd.DataFrame, base_period: str) -> pd.DataFrame:
    """Convert returns to 'one_period_len' rate of returns:
    that is the value the returns would have every 'one_period_len' if they
    had grown at a steady rate.

    Parameters
    ----------
    period_ret: pd.DataFrame
        A DataFrame containing returns values with column headings representing
        the return period.
    base_period: string
        The base period length used in the conversion
        It must follow pandas.Timedelta constructor format (e.g. '1 days',
        '1D', '30m', '3h', '1D1h', etc.)

    Returns
    -------
    pd.DataFrame
        A DataFrame in same format as input but with 'one_period_len' rate of
        returns values.
    """
    period_len = period_ret.name
    conversion_factor = pd.Timedelta(base_period) / pd.Timedelta(period_len)

    return period_ret.add(1).pow(conversion_factor).sub(1)


def std_conversion(period_std: pd.DataFrame, base_period: str) -> pd.DataFrame:
    """one_period_len standard deviation (or standard error) approximation.

    Parameters
    ----------
    period_std: pd.DataFrame
        A DataFrame containing standard deviation or standard error values
        with column headings representing the return period.
    base_period: string
        The base period length used in the conversion.
        It must follow pandas.Timedelta constructor format (e.g. '1 days',
        '1D', '30m', '3h', '1D1h', etc.)

    Returns
    -------
    pd.DataFrame
        A DataFrame in same format as input but with one-period
        standard deviation/error values.
    """
    period_len = period_std.name
    conversion_factor = pd.Timedelta(period_len) / pd.Timedelta(base_period)

    return period_std / np.sqrt(conversion_factor)


def get_forward_returns_columns(columns, require_exact_day_multiple=False):
    """
    Utility that detects and returns the columns that are forward returns.
    """

    # If exact day multiples are required in the forward return periods,
    # drop all other columns (e.g. drop 3D12h).
    pattern = re.compile(r"^(\d+([Dhms]|ms|us|ns]))+$", re.IGNORECASE)
    return_columns = [pattern.match(col) is not None for col in columns]
    if require_exact_day_multiple:
        pattern = re.compile(r"^(\d+(D))+$", re.IGNORECASE)
        valid_columns = [pattern.match(col) is not None for col in columns]
        if sum(valid_columns) < sum(return_columns):
            warnings.warn(
                "Skipping return periods that aren't exact multiples of days."
            )
    else:
        valid_columns = return_columns

    return columns[valid_columns]


def timedelta_to_string(timedelta: pd.Timedelta):
    """Utility that converts a pandas.Timedelta to a string representation
    compatible with pandas.Timedelta constructor format.

    Parameters
    ----------
    timedelta: pd.Timedelta

    Returns
    -------
    string
        string representation of 'timedelta'
    """
    c = timedelta.components
    fmt = ""
    if c.days != 0:
        fmt += "%dD" % c.days
    if c.hours > 0:
        fmt += "%dh" % c.hours
    if c.minutes > 0:
        fmt += "%dm" % c.minutes
    if c.seconds > 0:
        fmt += "%ds" % c.seconds
    if c.milliseconds > 0:
        fmt += "%dms" % c.milliseconds
    if c.microseconds > 0:
        fmt += "%dus" % c.microseconds
    if c.nanoseconds > 0:
        fmt += "%dns" % c.nanoseconds
    return fmt


def timedelta_strings_to_integers(sequence: Iterable) -> list:
    """Converts pandas string representations of timedelta into integers of
    days.

    Parameters
    ----------
    sequence : Iterable
        List or array of timedelta string representations, e.g. ['1D', '5D'].

    Returns
    -------
    sequence : list
        Integer days corresponding to the input sequence, e.g. [1, 5].
    """
    return list(map(lambda x: pd.Timedelta(x).days, sequence))


def add_custom_calendar_timedelta(
    in_put: pd.DatetimeIndex | pd.Timestamp,
    timedelta: pd.Timedelta,
    freq,
) -> pd.DatetimeIndex | pd.Timestamp:
    """Add timedelta to 'in_put' taking into consideration custom frequency,
    which is used to deal with custom calendars, such as a trading calendar.

    Parameters
    ----------
    in_put : pd.DatetimeIndex or pd.Timestamp
    timedelta : pd.Timedelta
    freq : pd.DataOffset (CustomBusinessDay, Day or BusinessDay)

    Returns
    -------
    pd.DatetimeIndex or pd.Timestamp
        input + timedelta
    """

    if not isinstance(freq, (Day, BDay, CDay)):
        raise ValueError("freq must be Day, BDay or CDay")

    days = timedelta.components.days
    offset = timedelta - pd.Timedelta(days=days)

    if isinstance(in_put, pd.DatetimeIndex):
        # Using vectoried operation in pandas for better performance.
        return in_put.map(lambda x: x + freq * days + offset)
    else:
        return in_put + freq * days + offset


def diff_custom_calendar_timedeltas(
    start: pd.Timestamp,
    end: pd.Timestamp,
    freq,
) -> pd.Timedelta:
    """Compute the difference between two pd.Timedelta taking into
    consideration custom frequency, which is used to deal with custom
    calendars, such as a trading calendar.

    Parameters
    ----------
    start : pd.Timestamp
    end : pd.Timestamp
    freq : pd.DateOffset (CDay, Day or BDay)

    # TODO: Remove following line.
    freq : CDay (see infer_trading_calendar)

    Returns
    -------
    pd.Timedelta
        end - start
    """

    if not isinstance(freq, (Day, BDay, CDay)):
        raise ValueError("freq must be Day, BDay or CDay")

    weekmask = getattr(freq, "weekmask", None)
    holidays = getattr(freq, "holidays", None)

    if weekmask is None and holidays is None:
        if isinstance(freq, Day):
            weekmask = "Mon Tue Wed Thu Fri Sat Sun"
            holidays = []
        elif isinstance(freq, BDay):
            weekmask = "Mon Tue Wed Thu Fri"
            holidays = []

    if weekmask is not None and holidays is not None:
        # We prefer this method as it is faster
        actual_days = np.busday_count(
            begindates=np.array(start.replace(tzinfo=None)).astype("datetime64[D]"),
            enddates=np.array(end.replace(tzinfo=None)).astype("datetime64[D]"),
            weekmask=weekmask,
            holidays=holidays,
        )
    else:
        # Default, it is slow
        actual_days = pd.date_range(start, end, freq=freq).shape[0] - 1
        if not freq.onOffset(start):
            actual_days -= 1

    time_diff = end - start
    delta_days = time_diff.components.days - actual_days

    return time_diff - pd.Timedelta(days=delta_days)


#################################################
# Following utility functions are from cofrics  #
#################################################
DATAREADER_DEPRECATION_WARNING = (
    "Yahoo and Google Finance have suffered large API breaks with no "
    "stable replacement. As a result, any data reading functionality "
    "in empyrical has been deprecated and will be removed in a future "
    "version. See README.md for more details: "
    "\n\n"
    "\thttps://github.com/quantopian/pyfolio/blob/master/README.md"
)


def roll(*args, **kwargs) -> np.ndarray | pd.Series:
    """Calculate a given statistics across a rolling time period.

    roll(returns, /, factor_returns=None, function, *, window, kwargs)

    Parameters
    ----------
    returns (positional): pd.Series | np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`.stats.cum_returns`.

    factor_returns (optional): float | pd.Series
        Benchmark return to compare returns against.

    function (keyword): callable, keyword argument
        the function to run for each rolling window.

    window (keyword): int
        the number of periods included in each calculation.

    (other keywords): other keywords that are required to be passed to
        the function in the 'function' argument may also be passed in.

    Returns
    -------
    np.ndarray | pd.Series
        depends on input type
        np.ndarray(s) ==> np.ndarray
        pd.Series(s) ==> pd.Series

        A Series or ndarray of the results of the stat across the rolling
        window.

    """
    func, window = kwargs.pop("function"), kwargs.pop("window")
    if len(args) > 2:
        raise ValueError("Cannot pass more than 2 return sets")
    elif len(args) == 2 and not isinstance(args[0], type(args[1])):
        raise ValueError("The two arguments types are not the same.")
    #  Choose type depends on the first one of args.
    elif isinstance(args[0], np.ndarray):
        return _roll_ndarray(func, window, *args, **kwargs)

    return _roll_pandas(func, window, *args, **kwargs)


def _roll_ndarray(func, window, *args, **kwargs) -> np.ndarray:
    data = []
    for i in range(window, len(args[0]) + 1):
        rets = [s[i - window : i] for s in args]
        data.append(func(*rets, **kwargs))

    return np.array(data)


def _roll_pandas(func, window, *args, **kwargs) -> pd.Series:
    data = {}
    index_values = []
    for i in range(window, len(args[0]) + 1):
        rets = [s.iloc[i - window : i] for s in args]
        index_value = args[0].index[i - 1]
        index_values.append(index_value)
        data[index_value] = func(*rets, **kwargs)

    return pd.Series(data, index=type(args[0].index)(index_values), dtype="float64")


def up(returns, factor_returns, **kwargs):
    """Calculates a given statistic filtering only positive factor return
    periods.

    Parameters
    ----------
    returns: pd.Series | np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~.stats.cum_returns`.

    factor_returns: (optional) float | pd.Series
        Benchmark return to compare returns against.

    function (keyword): callable
        The function to run for each rolling window.

    (other keywords): other keywords that are required to be passed to the
        function in the 'function' argument may also be passed in.

    Returns
    -------
    Same as the return of the function.
    """
    func = kwargs.pop("function")
    returns = returns[factor_returns > 0]
    factor_returns = factor_returns[factor_returns > 0]

    return func(returns, factor_returns, **kwargs)


def down(returns, factor_returns, **kwargs):
    """Calculates a given statistic filtering only negative factor return
    periods.

    Parameters
    ----------
    returns: pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~.stats.cum_returns`.

    factor_returns: (optional) float / series
        Benchmark return to compare returns against.

    function (keyword): callable
        the function to run for each rolling window.

    (other keywords): other keywords that are required to be passed to the
        function in the 'function' argument may also be passed in.

    Returns
    -------
    Same as the return of the 'function'
    """
    func = kwargs.pop("function")
    returns = returns[factor_returns < 0]
    factor_returns = factor_returns[factor_returns < 0]

    return func(returns, factor_returns, **kwargs)


def get_utc_timestamp(dt):
    """Returns the Timestamp/DatetimeIndex with either localized or converted
    to UTC

    Parameters
    ----------
    dt: Timestamp | DatetimeIndex
        the date(s) to be converted.

    Returns
    -------
    Same type as input
        date(s) converted to UTC
    """

    dt = pd.to_datetime(dt)
    try:
        dt = dt.tz_localize("UTC")
    except TypeError:
        dt = dt.tz_convert("UTC")

    return dt


_1_bday = BDay()


def _1_bday_ago():
    return pd.Timestamp.now().normalize() - _1_bday


def rolling_window(array: np.ndarray, length: int, mutable: bool = False) -> np.ndarray:
    """Restride an array of shape

        (X_0, X_1, ..., X_N)

    into an array of shape

        (X_0 - length + 1, length, X_1, ..., X_N)

    where each slice at index i along the first axis is equivalent to

        result[i] = array[i:i+length]

    Parameters
    ----------
    array: np.ndarray
        The base array.

    length: int
        Length of the synthetic first axis to generate.

    mutable: bool, optional
        Return a mutable array? The returned array shares the same memory as
        the input array. This means that writes into the returned array affect
        ``array``. The returned array also uses strides to map the same values
        to multiple indices. Writes to a single index may appear to change many
        values in the returned array.

    Returns
    -------
    out: np.ndarray

    Example
    -------
    >>> from numpy import arange
    >>> a = arange(25).reshape(5, 5)
    >>> a
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]])

    >>> rolling_window(a, 2)
    array([[[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9]],
    <BLANKLINE>
           [[ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14]],
    <BLANKLINE>
           [[10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19]],
    <BLANKLINE>
           [[15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24]]])
    """
    if not length:
        raise ValueError("Can't have 0-length window")

    orig_shape = array.shape
    if not orig_shape:
        raise IndexError("Can't restride a scalar.")
    elif orig_shape[0] < length:
        raise IndexError(
            f"Can't restride array of shape {orig_shape} with a window length of {length}"
        )

    num_windows = orig_shape[0] - length + 1
    new_shape = (num_windows, length) + orig_shape[1:]

    new_strides = (array.strides[0],) + array.strides

    out = as_strided(array, new_shape, new_strides)
    out.setflags(write=mutable)

    return out


# @deprecated(msg=DATAREADER_DEPRECATION_WARNING)
# def default_returns_func(symbol, start=None, end=None):
#     """Get returns for a symbol.

#     Queries Yahoo Finance. Attempts to cache SPY.

#     Parameters
#     ----------
#     symbol : object
#         Ticker symbol, e.g. 000001.SZ.
#     start : date, optional
#         Earliest date to fetch data for.
#         Defaults to the earliest date available.
#     end : date, optional
#         Latest date to fetch data for.
#         Defaults to latest date available.
#     Returns
#     -------
#     pd.Series
#         Daily returns for the symbol.
#          - See full explanation in tears.create_full_tear_sheet (returns).
#     """

#     if start is None:
#         start = "1/1/1970"
#     if end is None:
#         end = _1_bday_ago()

#     start = get_utc_timestamp(start)
#     end = get_utc_timestamp(end)

#     if symbol == "SPY":
#         filepath = data_path("spy.csv")
#         rets = get_returns_cached(
#             filepath,
#             get_symbol_returns_from_yahoo,
#             end,
#             symbol="SPY",
#             start="1/1/1970",
#             end=datetime.now(),
#         )
#         rets = rets[start:end]
#     else:
#         rets = get_symbol_returns_from_yahoo(symbol, start=start, end=end)

#     return rets[symbol]
