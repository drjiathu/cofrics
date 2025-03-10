from __future__ import division

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from cofrics.utils import (
    compute_forward_returns,
    get_clean_factor_and_forward_returns,
    quantize_factor,
)


class TestUtils:

    dr = pd.date_range(start="2015-01-01", end="2015-01-02")
    dr.name = "date"
    tickers = ["A", "B", "C", "D"]

    factor = pd.DataFrame(
        index=dr, columns=tickers, data=[[1, 2, 3, 4], [4, 3, 2, 1]]
    ).stack()
    factor.index = factor.index.set_names(["date", "asset"])
    factor.name = "factor"
    factor_data = pd.DataFrame()
    factor_data["factor"] = factor
    factor_data["group"] = pd.Series(
        index=factor.index, data=[1, 1, 2, 2, 1, 1, 2, 2], dtype="category"
    )

    biased_factor = pd.DataFrame(
        index=dr,
        columns=tickers.extend(["E", "F", "G", "H"]),
        data=[[-1, 3, -2, 4, -5, 7, -6, 8], [-4, 2, -3, 1, -8, 6, -7, 5]],
    ).stack()
    biased_factor.index = biased_factor.index.set_names(["date", "asset"])
    biased_factor.name = "factor"
    biased_factor_data = pd.DataFrame()
    biased_factor_data["factor"] = biased_factor
    biased_factor_data["group"] = pd.Series(
        index=biased_factor.index, data=[1, 1, 2, 2] * 4, dtype="category"
    )

    @staticmethod
    def test_compute_forward_returns():
        dr = pd.date_range(start="2015-1-1", end="2015-1-3")
        prices = pd.DataFrame(
            index=dr, columns=["A", "B"], data=[[1, 1], [1, 2], [2, 1]]
        )
        factor = prices.stack()

        fp = compute_forward_returns(factor, prices, periods=[1, 2])

        ix = pd.MultiIndex.from_product([dr, ["A", "B"]], names=["date", "asset"])
        expected = pd.DataFrame(index=ix, columns=["1D", "2D"])
        expected["1D"] = [0.0, 1.0, 1.0, -0.5, np.nan, np.nan]
        expected["2D"] = [1.0, 0.0, np.nan, np.nan, np.nan, np.nan]

        assert_frame_equal(fp, expected)

    @staticmethod
    def test_compute_forward_returns_index_out_of_bound():
        dr = pd.date_range(start="2014-12-29", end="2015-1-3")
        prices = pd.DataFrame(
            index=dr,
            columns=["A", "B"],
            data=[
                [np.nan, np.nan],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [1, 1],
                [1, 2],
                [2, 1],
            ],
        )
        dr = pd.date_range(start="2015-1-1", end="2015-1-3")
        factor = pd.DataFrame(
            index=dr, columns=["A", "B"], data=[[1, 1], [1, 2], [2, 1]]
        )
        factor = factor.stack()
        fp = compute_forward_returns(factor, prices, periods=[1, 2])
        ix = pd.MultiIndex.from_product([dr, ["A", "B"]], names=["date", "asset"])
        expected = pd.DataFrame(index=ix, columns=["1D", "2D"])
        expected["1D"] = [0.0, 1.0, 1.0, -0.5, np.nan, np.nan]
        expected["2D"] = [1.0, 0.0, np.nan, np.nan, np.nan, np.nan]

        assert_frame_equal(fp, expected)

    @staticmethod
    def test_compute_forward_returns_non_cum():
        dr = pd.date_range(start="2015-1-1", end="2015-1-3")
        prices = pd.DataFrame(
            index=dr, columns=["A", "B"], data=[[1, 1], [1, 2], [2, 1]]
        )
        factor = prices.stack()
        fp = compute_forward_returns(
            factor, prices, periods=[1, 2], cumulative_returns=False
        )
        ix = pd.MultiIndex.from_product([dr, ["A", "B"]], names=["date", "asset"])
        expected = pd.DataFrame(index=ix, columns=["1D", "2D"])
        expected["1D"] = [0.0, 1.0, 1.0, -0.5, np.nan, np.nan]
        expected["2D"] = [1.0, -0.5, np.nan, np.nan, np.nan, np.nan]

        assert_frame_equal(fp, expected)

    @pytest.mark.parametrize(
        "factor, quantiles, bins, by_group, zero_aware, expected_vals",
        [
            (factor_data, 4, None, False, False, [1, 2, 3, 4, 4, 3, 2, 1]),
            (factor_data, 2, None, False, False, [1, 1, 2, 2, 2, 2, 1, 1]),
            (factor_data, 2, None, True, False, [1, 2, 1, 2, 2, 1, 2, 1]),
            (
                biased_factor_data,
                4,
                None,
                False,
                True,
                [2, 3, 2, 3, 1, 4, 1, 4, 2, 3, 2, 3, 1, 4, 1, 4],
            ),
            (
                biased_factor_data,
                2,
                None,
                False,
                True,
                [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            ),
            (
                biased_factor_data,
                2,
                None,
                True,
                True,
                [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            ),
            (
                biased_factor_data,
                None,
                4,
                False,
                True,
                [2, 3, 2, 3, 1, 4, 1, 4, 2, 3, 2, 3, 1, 4, 1, 4],
            ),
            (
                biased_factor_data,
                None,
                2,
                False,
                True,
                [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            ),
            (
                biased_factor_data,
                None,
                2,
                True,
                True,
                [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            ),
            (
                factor_data,
                [0, 0.25, 0.5, 0.75, 1.0],
                None,
                False,
                False,
                [1, 2, 3, 4, 4, 3, 2, 1],
            ),
            (
                factor_data,
                [0, 0.5, 0.75, 1.0],
                None,
                False,
                False,
                [1, 1, 2, 3, 3, 2, 1, 1],
            ),
            (
                factor_data,
                [0, 0.25, 0.5, 1.0],
                None,
                False,
                False,
                [1, 2, 3, 3, 3, 3, 2, 1],
            ),
            (factor_data, [0, 0.5, 1.0], None, False, False, [1, 1, 2, 2, 2, 2, 1, 1]),
            (
                factor_data,
                [0.25, 0.5, 0.75],
                None,
                False,
                False,
                [np.nan, 1, 2, np.nan, np.nan, 2, 1, np.nan],
            ),
            (factor_data, [0, 0.5, 1.0], None, True, False, [1, 2, 1, 2, 2, 1, 2, 1]),
            (
                factor_data,
                [0.5, 1.0],
                None,
                True,
                False,
                [np.nan, 1, np.nan, 1, 1, np.nan, 1, np.nan],
            ),
            (factor_data, [0, 1.0], None, True, False, [1, 1, 1, 1, 1, 1, 1, 1]),
            (factor_data, None, 4, False, False, [1, 2, 3, 4, 4, 3, 2, 1]),
            (factor_data, None, 2, False, False, [1, 1, 2, 2, 2, 2, 1, 1]),
            (factor_data, None, 3, False, False, [1, 1, 2, 3, 3, 2, 1, 1]),
            (factor_data, None, 8, False, False, [1, 3, 6, 8, 8, 6, 3, 1]),
            (
                factor_data,
                None,
                [0, 1, 2, 3, 5],
                False,
                False,
                [1, 2, 3, 4, 4, 3, 2, 1],
            ),
            (
                factor_data,
                None,
                [1, 2, 3],
                False,
                False,
                [np.nan, 1, 2, np.nan, np.nan, 2, 1, np.nan],
            ),
            (factor_data, None, [0, 2, 5], False, False, [1, 1, 2, 2, 2, 2, 1, 1]),
            (
                factor_data,
                None,
                [0.5, 2.5, 4.5],
                False,
                False,
                [1, 1, 2, 2, 2, 2, 1, 1],
            ),
            (
                factor_data,
                None,
                [0.5, 2.5],
                True,
                False,
                [1, 1, np.nan, np.nan, np.nan, np.nan, 1, 1],
            ),
            (factor_data, None, 2, True, False, [1, 2, 1, 2, 2, 1, 2, 1]),
        ],
    )
    def test_quantize_factor(
        self, factor, quantiles, bins, by_group, zero_aware, expected_vals
    ):

        quantized_factor = quantize_factor(
            factor_data=factor,
            quantiles=quantiles,
            bins=bins,
            by_group=by_group,
            zero_aware=zero_aware,
        )
        expected = pd.Series(
            index=factor.index, data=expected_vals, name="factor_quantile"
        ).dropna()

        assert_series_equal(quantized_factor, expected)

    @staticmethod
    def test_get_clean_factor_and_forward_returns_0():
        """
        Test get_clean_factor_and_forward_returns with a daily factor.
        """
        tickers = ["A", "B", "C", "D", "E", "F"]
        factor_groups = {"A": 1, "B": 2, "C": 1, "D": 2, "E": 1, "F": 2}
        price_data = [
            [1.10**i, 0.50**i, 3.00**i, 0.90**i, 0.50**i, 1.00**i] for i in range(1, 7)
        ]  # 6 days = 3 + 3 fwd returns
        factor_data = [
            [3, 4, 2, 1, np.nan, np.nan],
            [3, np.nan, np.nan, 1, 4, 2],
            [3, 4, 2, 1, np.nan, np.nan],
        ]  # 3 days

        start = "2015-1-11"
        factor_end = "2015-1-13"
        price_end = "2015-1-16"  # 3D fwd returns

        price_index = pd.date_range(start=start, end=price_end)
        price_index.name = "date"
        prices = pd.DataFrame(index=price_index, columns=tickers, data=price_data)
        factor_index = pd.date_range(start=start, end=factor_end)
        factor_index.name = "date"
        factor = pd.DataFrame(
            index=factor_index, columns=tickers, data=factor_data
        ).stack()

        factor_data = get_clean_factor_and_forward_returns(
            factor, prices, groupby=factor_groups, quantiles=4, periods=(1, 2, 3)
        )

        expected_idx = factor.index.rename(["date", "asset"])
        expected_cols = ["1D", "2D", "3D", "factor", "group", "factor_quantile"]
        expected_data = [
            [0.1, 0.21, 0.331, 3.0, 1, 3],
            [-0.5, -0.75, -0.875, 4.0, 2, 4],
            [2.0, 8.00, 26.000, 2.0, 1, 2],
            [-0.1, -0.19, -0.271, 1.0, 2, 1],
            [0.1, 0.21, 0.331, 3.0, 1, 3],
            [-0.1, -0.19, -0.271, 1.0, 2, 1],
            [-0.5, -0.75, -0.875, 4.0, 1, 4],
            [0.0, 0.00, 0.000, 2.0, 2, 2],
            [0.1, 0.21, 0.331, 3.0, 1, 3],
            [-0.5, -0.75, -0.875, 4.0, 2, 4],
            [2.0, 8.00, 26.000, 2.0, 1, 2],
            [-0.1, -0.19, -0.271, 1.0, 2, 1],
        ]
        expected = pd.DataFrame(
            index=expected_idx, columns=expected_cols, data=expected_data
        )
        expected["group"] = expected["group"].astype("category")

        assert_frame_equal(factor_data, expected)

    @staticmethod
    def test_get_clean_factor_and_forward_returns_1():
        """Test get_clean_factor_and_forward_returns with a daily factor on a
        business day calendar.
        """
        tickers = ["A", "B", "C", "D", "E", "F"]

        factor_groups = {"A": 1, "B": 2, "C": 1, "D": 2, "E": 1, "F": 2}

        price_data = [
            [1.10**i, 0.50**i, 3.00**i, 0.90**i, 0.50**i, 1.00**i] for i in range(1, 7)
        ]  # 6 days = 3 + 3 fwd returns

        factor_data = [
            [3, 4, 2, 1, np.nan, np.nan],
            [3, np.nan, np.nan, 1, 4, 2],
            [3, 4, 2, 1, np.nan, np.nan],
        ]  # 3 days

        start = "2017-1-12"
        factor_end = "2017-1-16"
        price_end = "2017-1-19"  # 3D fwd returns
        price_index = pd.date_range(start=start, end=price_end, freq="B")
        price_index.name = "date"
        prices = pd.DataFrame(index=price_index, columns=tickers, data=price_data)

        factor_index = pd.date_range(start=start, end=factor_end, freq="B")
        factor_index.name = "date"
        factor = pd.DataFrame(
            index=factor_index, columns=tickers, data=factor_data
        ).stack()
        factor_data = get_clean_factor_and_forward_returns(
            factor, prices, groupby=factor_groups, quantiles=4, periods=(1, 2, 3)
        )

        expected_idx = factor.index.rename(["date", "asset"])
        expected_cols = ["1D", "2D", "3D", "factor", "group", "factor_quantile"]
        expected_data = [
            [0.1, 0.21, 0.331, 3.0, 1, 3],
            [-0.5, -0.75, -0.875, 4.0, 2, 4],
            [2.0, 8.00, 26.000, 2.0, 1, 2],
            [-0.1, -0.19, -0.271, 1.0, 2, 1],
            [0.1, 0.21, 0.331, 3.0, 1, 3],
            [-0.1, -0.19, -0.271, 1.0, 2, 1],
            [-0.5, -0.75, -0.875, 4.0, 1, 4],
            [0.0, 0.00, 0.000, 2.0, 2, 2],
            [0.1, 0.21, 0.331, 3.0, 1, 3],
            [-0.5, -0.75, -0.875, 4.0, 2, 4],
            [2.0, 8.00, 26.000, 2.0, 1, 2],
            [-0.1, -0.19, -0.271, 1.0, 2, 1],
        ]
        expected = pd.DataFrame(
            index=expected_idx, columns=expected_cols, data=expected_data
        )
        expected["group"] = expected["group"].astype("category")

        assert_frame_equal(factor_data, expected)

    @staticmethod
    def test_get_clean_factor_and_forward_returns_2():
        """
        Test get_clean_factor_and_forward_returns with and intraday factor
        """
        tickers = ["A", "B", "C", "D", "E", "F"]
        factor_groups = {"A": 1, "B": 2, "C": 1, "D": 2, "E": 1, "F": 2}
        price_data = [
            [1.10**i, 0.50**i, 3.00**i, 0.90**i, 0.50**i, 1.00**i] for i in range(1, 5)
        ]  # 4 days = 3 + 1 fwd returns
        factor_data = [
            [3, 4, 2, 1, np.nan, np.nan],
            [3, np.nan, np.nan, 1, 4, 2],
            [3, 4, 2, 1, np.nan, np.nan],
        ]  # 3 days

        start = "2017-1-12"
        factor_end = "2017-1-16"
        price_end = "2017-1-17"  # 1D fwd returns

        price_index = pd.date_range(start=start, end=price_end, freq="B")
        price_index.name = "date"
        today_open = pd.DataFrame(
            index=price_index + pd.Timedelta("9h30m"),
            columns=tickers,
            data=price_data,
        )
        today_open_1h = pd.DataFrame(
            index=price_index + pd.Timedelta("10h30m"),
            columns=tickers,
            data=price_data,
        )
        today_open_1h += today_open_1h * 0.001
        today_open_3h = pd.DataFrame(
            index=price_index + pd.Timedelta("12h30m"),
            columns=tickers,
            data=price_data,
        )
        today_open_3h -= today_open_3h * 0.002
        prices = pd.concat([today_open, today_open_1h, today_open_3h]).sort_index()

        factor_index = pd.date_range(start=start, end=factor_end, freq="B")
        factor_index.name = "date"
        factor = pd.DataFrame(
            index=factor_index + pd.Timedelta("9h30m"),
            columns=tickers,
            data=factor_data,
        ).stack()

        factor_data = get_clean_factor_and_forward_returns(
            factor, prices, groupby=factor_groups, quantiles=4, periods=(1, 2, 3)
        )

        expected_idx = factor.index.rename(["date", "asset"])
        expected_cols = ["1h", "3h", "1D", "factor", "group", "factor_quantile"]
        expected_data = [
            [0.001, -0.002, 0.1, 3.0, 1, 3],
            [0.001, -0.002, -0.5, 4.0, 2, 4],
            [0.001, -0.002, 2.0, 2.0, 1, 2],
            [0.001, -0.002, -0.1, 1.0, 2, 1],
            [0.001, -0.002, 0.1, 3.0, 1, 3],
            [0.001, -0.002, -0.1, 1.0, 2, 1],
            [0.001, -0.002, -0.5, 4.0, 1, 4],
            [0.001, -0.002, 0.0, 2.0, 2, 2],
            [0.001, -0.002, 0.1, 3.0, 1, 3],
            [0.001, -0.002, -0.5, 4.0, 2, 4],
            [0.001, -0.002, 2.0, 2.0, 1, 2],
            [0.001, -0.002, -0.1, 1.0, 2, 1],
        ]
        expected = pd.DataFrame(
            index=expected_idx, columns=expected_cols, data=expected_data
        )
        expected["group"] = expected["group"].astype("category")

        assert_frame_equal(factor_data, expected)

    @staticmethod
    def test_get_clean_factor_and_forward_returns_3():
        """
        Test get_clean_factor_and_forward_returns on an event
        """
        tickers = ["A", "B", "C", "D", "E", "F"]

        factor_groups = {"A": 1, "B": 2, "C": 1, "D": 2, "E": 1, "F": 2}

        price_data = [
            [1.10**i, 0.50**i, 3.00**i, 0.90**i, 0.50**i, 1.00**i] for i in range(1, 9)
        ]

        factor_data = [
            [1, np.nan, np.nan, np.nan, np.nan, 6],
            [4, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, 3, np.nan, 2, np.nan, np.nan],
            [np.nan, np.nan, 1, np.nan, 3, np.nan],
        ]

        price_index = pd.date_range(start="2017-1-12", end="2017-1-23", freq="B")
        price_index.name = "date"
        prices = pd.DataFrame(index=price_index, columns=tickers, data=price_data)
        factor_index = pd.date_range(start="2017-1-12", end="2017-1-18", freq="B")
        factor_index.name = "date"
        factor = pd.DataFrame(
            index=factor_index, columns=tickers, data=factor_data
        ).stack()
        factor_data = get_clean_factor_and_forward_returns(
            factor, prices, groupby=factor_groups, quantiles=4, periods=(1, 2, 3)
        )

        expected_idx = factor.index.rename(["date", "asset"])
        expected_cols = ["1D", "2D", "3D", "factor", "group", "factor_quantile"]
        expected_data = [
            [0.1, 0.21, 0.331, 1.0, 1, 1],
            [0.0, 0.00, 0.000, 6.0, 2, 4],
            [0.1, 0.21, 0.331, 4.0, 1, 1],
            [-0.1, -0.19, -0.271, 7.0, 2, 4],
            [-0.5, -0.75, -0.875, 3.0, 2, 4],
            [-0.1, -0.19, -0.271, 2.0, 2, 1],
            [2.0, 8.00, 26.000, 1.0, 1, 1],
            [-0.5, -0.75, -0.875, 3.0, 1, 4],
        ]
        expected = pd.DataFrame(
            index=expected_idx, columns=expected_cols, data=expected_data
        )
        expected["group"] = expected["group"].astype("category")

        assert_frame_equal(factor_data, expected)

    def test_get_clean_factor_and_forward_returns_4(self):
        """
        Test get_clean_factor_and_forward_returns with and intraday factor
        and holidays
        """
        tickers = ["A", "B", "C", "D", "E", "F"]
        factor_groups = {"A": 1, "B": 2, "C": 1, "D": 2, "E": 1, "F": 2}
        price_data = [
            [1.10**i, 0.50**i, 3.00**i, 0.90**i, 0.50**i, 1.00**i] for i in range(1, 20)
        ]  # 19 days = 18 + 1 fwd returns

        factor_data = [
            [3, 4, 2, 1, np.nan, np.nan],
            [3, np.nan, np.nan, 1, 4, 2],
            [3, 4, 2, 1, np.nan, np.nan],
        ] * 6  # 18 days

        start = "2017-1-12"
        factor_end = "2017-2-10"
        price_end = "2017-2-13"  # 1D (business day) fwd returns
        holidays = ["2017-1-13", "2017-1-18", "2017-1-30", "2017-2-7"]
        holidays = [pd.Timestamp(d) for d in holidays]

        price_index = pd.date_range(start=start, end=price_end, freq="B")
        price_index.name = "date"
        price_index = price_index.drop(holidays)

        today_open = pd.DataFrame(
            index=price_index + pd.Timedelta("9h30m"),
            columns=tickers,
            data=price_data,
        )
        today_open_1h = pd.DataFrame(
            index=price_index + pd.Timedelta("10h30m"),
            columns=tickers,
            data=price_data,
        )
        today_open_1h += today_open_1h * 0.001
        today_open_3h = pd.DataFrame(
            index=price_index + pd.Timedelta("12h30m"), columns=tickers, data=price_data
        )
        today_open_3h -= today_open_3h * 0.002
        prices = pd.concat([today_open, today_open_1h, today_open_3h]).sort_index()

        factor_index = pd.date_range(start=start, end=factor_end, freq="B")
        factor_index.name = "date"
        factor_index = factor_index.drop(holidays)
        factor = pd.DataFrame(
            index=factor_index + pd.Timedelta("9h30m"),
            columns=tickers,
            data=factor_data,
        ).stack()

        factor_data = get_clean_factor_and_forward_returns(
            factor, prices, groupby=factor_groups, quantiles=4, periods=(1, 2, 3)
        )

        expected_idx = factor.index.rename(["date", "asset"])
        expected_cols = ["1h", "3h", "1D", "factor", "group", "factor_quantile"]
        expected_data = [
            [0.001, -0.002, 0.1, 3.0, 1, 3],
            [0.001, -0.002, -0.5, 4.0, 2, 4],
            [0.001, -0.002, 2.0, 2.0, 1, 2],
            [0.001, -0.002, -0.1, 1.0, 2, 1],
            [0.001, -0.002, 0.1, 3.0, 1, 3],
            [0.001, -0.002, -0.1, 1.0, 2, 1],
            [0.001, -0.002, -0.5, 4.0, 1, 4],
            [0.001, -0.002, 0.0, 2.0, 2, 2],
            [0.001, -0.002, 0.1, 3.0, 1, 3],
            [0.001, -0.002, -0.5, 4.0, 2, 4],
            [0.001, -0.002, 2.0, 2.0, 1, 2],
            [0.001, -0.002, -0.1, 1.0, 2, 1],
        ] * 6  # 18 days
        expected = pd.DataFrame(
            index=expected_idx, columns=expected_cols, data=expected_data
        )
        expected["group"] = expected["group"].astype("category")

        assert_frame_equal(factor_data, expected)

        inferred_holidays = factor_data.index.levels[0].freq.holidays
        assert sorted(holidays) == sorted(inferred_holidays)

    def test_get_clean_factor_and_forward_returns_5(self):
        """
        Test get_clean_factor_and_forward_returns with a daily factor
        on a business day calendar and holidays
        """
        tickers = ["A", "B", "C", "D", "E", "F"]

        factor_groups = {"A": 1, "B": 2, "C": 1, "D": 2, "E": 1, "F": 2}

        price_data = [
            [1.10**i, 0.50**i, 3.00**i, 0.90**i, 0.50**i, 1.00**i] for i in range(1, 22)
        ]  # 21 days = 18 + 3 fwd returns

        factor_data = [
            [3, 4, 2, 1, np.nan, np.nan],
            [3, np.nan, np.nan, 1, 4, 2],
            [3, 4, 2, 1, np.nan, np.nan],
        ] * 6  # 18 days

        start = "2017-1-12"
        factor_end = "2017-2-10"
        price_end = "2017-2-15"  # 3D (business day) fwd returns
        holidays = ["2017-1-13", "2017-1-18", "2017-1-30", "2017-2-7"]
        holidays = [pd.Timestamp(d) for d in holidays]

        price_index = pd.date_range(start=start, end=price_end, freq="B")
        price_index.name = "date"
        price_index = price_index.drop(holidays)
        prices = pd.DataFrame(index=price_index, columns=tickers, data=price_data)

        factor_index = pd.date_range(start=start, end=factor_end, freq="B")
        factor_index.name = "date"
        factor_index = factor_index.drop(holidays)
        factor = pd.DataFrame(
            index=factor_index, columns=tickers, data=factor_data
        ).stack()

        factor_data = get_clean_factor_and_forward_returns(
            factor, prices, groupby=factor_groups, quantiles=4, periods=(1, 2, 3)
        )
        expected_idx = factor.index.rename(["date", "asset"])
        expected_cols = ["1D", "2D", "3D", "factor", "group", "factor_quantile"]
        expected_data = [
            [0.1, 0.21, 0.331, 3.0, 1, 3],
            [-0.5, -0.75, -0.875, 4.0, 2, 4],
            [2.0, 8.00, 26.000, 2.0, 1, 2],
            [-0.1, -0.19, -0.271, 1.0, 2, 1],
            [0.1, 0.21, 0.331, 3.0, 1, 3],
            [-0.1, -0.19, -0.271, 1.0, 2, 1],
            [-0.5, -0.75, -0.875, 4.0, 1, 4],
            [0.0, 0.00, 0.000, 2.0, 2, 2],
            [0.1, 0.21, 0.331, 3.0, 1, 3],
            [-0.5, -0.75, -0.875, 4.0, 2, 4],
            [2.0, 8.00, 26.000, 2.0, 1, 2],
            [-0.1, -0.19, -0.271, 1.0, 2, 1],
        ] * 6  # 18  days
        expected = pd.DataFrame(
            index=expected_idx, columns=expected_cols, data=expected_data
        )
        expected["group"] = expected["group"].astype("category")

        assert_frame_equal(factor_data, expected)

        inferred_holidays = factor_data.index.levels[0].freq.holidays

        assert sorted(holidays) == sorted(inferred_holidays)
