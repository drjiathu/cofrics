from __future__ import annotations, division

from copy import copy
from functools import wraps
from operator import attrgetter

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_almost_equal
from pandas.core.generic import NDFrame
from pandas.testing import assert_index_equal
from scipy import stats

import cofrics
from cofrics.periods import DAILY, MONTHLY, QUARTERLY, WEEKLY, YEARLY
from cofrics.stats import alpha_aligned, capture, cum_returns_final, max_drawdown
from cofrics.utils import down, roll, up

DECIMAL_PLACES = 8

# NOTE: Don't change seed since some tests rely on it.
rand = np.random.RandomState(1337)


class TestBaseCase:
    @staticmethod
    def assert_indexes_match(result, expected):
        """Assert that two pandas objects have the same indices.

        This is a method instead of a free function so that we can override it
        to be a no-op in suites like TestStatsArrays that unwrap pandas objects
        into np.ndarray.

        Parameters
        ----------
        result :

        expected :

        Returns
        -------

        """
        assert_index_equal(result.index, expected.index)

        if isinstance(result, pd.DataFrame) and isinstance(expected, pd.DataFrame):
            assert_index_equal(result.columns, expected.columns)


class TestStats(TestBaseCase):
    # Simple benchmark, no drawdown
    simple_benchmark = pd.Series(
        np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]) / 100,
        index=pd.date_range("2000-1-30", periods=9, freq="D"),
    )

    # All positive returns, small variance
    positive_returns = pd.Series(
        np.array([1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) / 100,
        index=pd.date_range("2000-1-30", periods=9, freq="D"),
    )

    # All negative returns
    negative_returns = pd.Series(
        np.array([0.0, -6.0, -7.0, -1.0, -9.0, -2.0, -6.0, -8.0, -5.0]) / 100,
        index=pd.date_range("2000-1-30", periods=9, freq="D"),
    )

    # All negative returns
    all_negative_returns = pd.Series(
        np.array([-2.0, -6.0, -7.0, -1.0, -9.0, -2.0, -6.0, -8.0, -5.0]) / 100,
        index=pd.date_range("2000-1-30", periods=9, freq="D"),
    )

    # Positive and negative returns with max drawdown
    mixed_returns = pd.Series(
        np.array([np.nan, 1.0, 10.0, -4.0, 2.0, 3.0, 2.0, 1.0, -10.0]) / 100,
        index=pd.date_range("2000-1-30", periods=9, freq="D"),
    )

    # Flat line
    flat_line_1 = pd.Series(
        np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) / 100,
        index=pd.date_range("2000-1-30", periods=9, freq="D"),
    )

    # Weekly returns
    weekly_returns = pd.Series(
        np.array([0.0, 1.0, 10.0, -4.0, 2.0, 3.0, 2.0, 1.0, -10.0]) / 100,
        index=pd.date_range("2000-1-30", periods=9, freq="W"),
    )

    # Monthly returns
    monthly_returns = pd.Series(
        np.array([0.0, 1.0, 10.0, -4.0, 2.0, 3.0, 2.0, 1.0, -10.0]) / 100,
        index=pd.date_range("2000-1-30", periods=9, freq="ME"),
    )

    # Series of length 1
    one_return = pd.Series(
        np.array([1.0]) / 100, index=pd.date_range("2000-1-30", periods=1, freq="D")
    )

    # Empty series
    empty_returns = pd.Series(
        np.array([]) / 100, index=pd.date_range("2000-1-30", periods=0, freq="D")
    )

    # Random noise
    noise = pd.Series(
        rand.normal(0, 0.001, 1000),
        index=pd.date_range("2000-1-30", periods=1000, freq="D", tz="UTC"),
    )

    noise_uniform = pd.Series(
        rand.uniform(-0.01, 0.01, 1000),
        index=pd.date_range("2000-1-30", periods=1000, freq="D", tz="UTC"),
    )

    # Random noise inv
    inv_noise = noise.multiply(-1)

    # Flat line
    flat_line_0 = pd.Series(
        np.linspace(0, 0, num=1000),
        index=pd.date_range("2000-1-30", periods=1000, freq="D", tz="UTC"),
    )
    # Flat line
    flat_line_1_tz = pd.Series(
        np.linspace(0.01, 0.01, num=1000),
        index=pd.date_range("2000-1-30", periods=1000, freq="D", tz="UTC"),
    )

    # Positive line
    pos_line = pd.Series(
        np.linspace(0, 1, num=1000),
        index=pd.date_range("2000-1-30", periods=1000, freq="D", tz="UTC"),
    )

    # Negative line
    neg_line = pd.Series(
        np.linspace(0, -1, num=1000),
        index=pd.date_range("2000-1-30", periods=1000, freq="D", tz="UTC"),
    )

    # Sparse noise, same as noise but with np.nan sprinkled in
    replace_nan = rand.choice(noise.index.tolist(), rand.randint(1, 10))
    sparse_noise = noise
    sparse_noise[sparse_noise.index.isin(replace_nan)] = np.nan

    # Sparse flat line at 0.01
    replace_nan = rand.choice(noise.index.tolist(), rand.randint(1, 10))
    sparse_flat_line_1_tz = flat_line_1_tz.replace(replace_nan, np.nan)

    one = [
        -0.00171614,
        0.01322056,
        0.03063862,
        -0.01422057,
        -0.00489779,
        0.01268925,
        -0.03357711,
        0.01797036,
    ]
    two = [
        0.01846232,
        0.00793951,
        -0.01448395,
        0.00422537,
        -0.00339611,
        0.03756813,
        0.0151531,
        0.03549769,
    ]

    df_index_simple = pd.date_range("2000-1-30", periods=8, freq="D")
    df_index_week = pd.date_range("2000-1-30", periods=8, freq="W")
    df_index_month = pd.date_range("2000-1-30", periods=8, freq="ME")

    df_simple = pd.DataFrame(
        {
            "one": pd.Series(one, index=df_index_simple),
            "two": pd.Series(two, index=df_index_simple),
        }
    )

    df_week = pd.DataFrame(
        {
            "one": pd.Series(one, index=df_index_week),
            "two": pd.Series(two, index=df_index_week),
        }
    )

    df_month = pd.DataFrame(
        {
            "one": pd.Series(one, index=df_index_month),
            "two": pd.Series(two, index=df_index_month),
        }
    )

    @pytest.mark.parametrize(
        "prices, expected",
        [
            # Constant price implies zero returns,
            # and linearly increasing prices implies returns like 1/n
            (flat_line_1, [0.0] * (flat_line_1.shape[0] - 1)),
            (pos_line, [np.inf] + [1 / n for n in range(1, 999)]),
        ],
    )
    def test_simple_returns(self, prices, expected):
        simple_returns = self.cofrics.simple_returns(prices)
        assert_almost_equal(np.array(simple_returns), expected, 4)
        self.assert_indexes_match(simple_returns, prices.iloc[1:])

    @pytest.mark.parametrize(
        "returns, starting_value, expected",
        [
            (empty_returns, 0, []),
            (
                mixed_returns,
                0,
                [
                    0.0,
                    0.01,
                    0.111,
                    0.066559,
                    0.08789,
                    0.12052,
                    0.14293,
                    0.15436,
                    0.03893,
                ],
            ),
            (
                mixed_returns,
                100,
                [
                    100.0,
                    101.0,
                    111.1,
                    106.65599,
                    108.78912,
                    112.05279,
                    114.29384,
                    115.43678,
                    103.89310,
                ],
            ),
            (
                negative_returns,
                0,
                [
                    0.0,
                    -0.06,
                    -0.1258,
                    -0.13454,
                    -0.21243,
                    -0.22818,
                    -0.27449,
                    -0.33253,
                    -0.36590,
                ],
            ),
        ],
    )
    def test_cum_returns(self, returns, starting_value, expected):
        cum_returns = self.cofrics.cum_returns(
            returns,
            starting_value=starting_value,
        )
        for i in range(returns.size):
            if isinstance(cum_returns, np.ndarray):
                assert_almost_equal(cum_returns[i], expected[i], 4)
            else:
                assert_almost_equal(cum_returns.iloc[i], expected[i], 4)
        self.assert_indexes_match(cum_returns, returns)

    @pytest.mark.parametrize(
        "returns, starting_value, expected",
        [
            (empty_returns, 0, np.nan),
            (one_return, 0, one_return.iloc[0]),
            (mixed_returns, 0, 0.03893),
            (mixed_returns, 100, 103.89310),
            (negative_returns, 0, -0.36590),
        ],
    )
    def test_cum_returns_final(self, returns, starting_value, expected):
        cum_returns_final = self.cofrics.cum_returns_final(
            returns,
            starting_value=starting_value,
        )
        assert_almost_equal(cum_returns_final, expected, 4)

    @pytest.mark.parametrize(
        "returns, convert_to, expected",
        [
            (simple_benchmark, WEEKLY, [0.0, 0.040604010000000024, 0.0]),
            (simple_benchmark, MONTHLY, [0.01, 0.03030099999999991]),
            (simple_benchmark, QUARTERLY, [0.04060401]),
            (simple_benchmark, YEARLY, [0.040604010000000024]),
            (
                weekly_returns,
                MONTHLY,
                [0.0, 0.087891200000000058, -0.04500459999999995],
            ),
            (weekly_returns, YEARLY, [0.038931091700480147]),
            (monthly_returns, YEARLY, [0.038931091700480147]),
            (
                monthly_returns,
                QUARTERLY,
                [0.11100000000000021, 0.008575999999999917, -0.072819999999999996],
            ),
        ],
    )
    def test_aggregate_returns(self, returns, convert_to, expected):
        returns = (
            self.cofrics(pandas_only=True)
            .aggregate_returns(returns, convert_to)
            .values.tolist()
        )

        for i, v in enumerate(returns):
            assert_almost_equal(v, expected[i], DECIMAL_PLACES)

    @pytest.mark.parametrize(
        "returns, expected",
        [
            (empty_returns, np.nan),
            (one_return, 0.0),
            (simple_benchmark, 0.0),
            (mixed_returns, -0.1),
            (positive_returns, -0.0),
            # negative returns means the drawdown is just the returns
            (negative_returns, cum_returns_final(negative_returns)),
            (all_negative_returns, cum_returns_final(all_negative_returns)),
            (
                pd.Series(
                    np.array([10, -10, 10]) / 100,
                    index=pd.date_range("2000-1-30", periods=3, freq="D"),
                ),
                -0.10,
            ),
        ],
    )
    def test_max_drawdown(self, returns, expected):
        assert_almost_equal(
            self.cofrics.max_drawdown(returns), expected, DECIMAL_PLACES
        )

    # Maximum drawdown is always less than or equal to zero. Translating
    # returns by a positive constant should increase the maximum
    # drawdown to a maximum of zero. Translating by a negative constant
    # decreases the maximum drawdown.
    @pytest.mark.parametrize(
        "returns, constant",
        [
            (noise, 0.0001),
            (noise, 0.001),
            (noise_uniform, 0.01),
            (noise_uniform, 0.1),
        ],
    )
    def test_max_drawdown_translation(self, returns, constant):
        depressed_returns = returns - constant
        raised_returns = returns + constant

        max_dd = self.cofrics.max_drawdown(returns)
        depressed_dd = self.cofrics.max_drawdown(depressed_returns)
        raised_dd = self.cofrics.max_drawdown(raised_returns)

        assert max_dd <= raised_dd
        assert depressed_dd <= max_dd

    @pytest.mark.parametrize(
        "returns, period, expected",
        [
            (mixed_returns, DAILY, 1.9135925373194231),
            (weekly_returns, WEEKLY, 0.24690830513998208),
            (monthly_returns, MONTHLY, 0.052242061386048144),
        ],
    )
    def test_annual_ret(self, returns, period, expected):
        assert_almost_equal(
            self.cofrics.annual_return(returns, period=period), expected, DECIMAL_PLACES
        )

    @pytest.mark.parametrize(
        "returns, period, expected",
        [
            (flat_line_1_tz, DAILY, 0.0),
            (mixed_returns, DAILY, 0.9136465399704637),
            (weekly_returns, WEEKLY, 0.38851569394870583),
            (monthly_returns, MONTHLY, 0.18663690238892558),
        ],
    )
    def test_annual_volatility(self, returns, period, expected):
        assert_almost_equal(
            self.cofrics.annual_volatility(returns, period=period),
            expected,
            DECIMAL_PLACES,
        )

    @pytest.mark.parametrize(
        "returns, period, expected",
        [
            (empty_returns, DAILY, np.nan),
            (one_return, DAILY, np.nan),
            (mixed_returns, DAILY, 19.135925373194233),
            (weekly_returns, WEEKLY, 2.4690830513998208),
            (monthly_returns, MONTHLY, 0.52242061386048144),
        ],
    )
    def test_calmar(self, returns, period, expected):
        assert_almost_equal(
            self.cofrics.calmar_ratio(returns, period=period), expected, DECIMAL_PLACES
        )

    # Regression tests for omega ratio.
    @pytest.mark.parametrize(
        "returns, risk_free, required_return, expected",
        [
            (empty_returns, 0.0, 0.0, np.nan),
            (one_return, 0.0, 0.0, np.nan),
            (mixed_returns, 0.0, 10.0, 0.83354263497557934),
            (mixed_returns, 0.0, -10.0, np.nan),
            (mixed_returns, flat_line_1, 0.0, 0.8125),
            (positive_returns, 0.01, 0.0, np.nan),
            (positive_returns, 0.011, 0.0, 1.125),
            (positive_returns, 0.02, 0.0, 0.0),
            (negative_returns, 0.01, 0.0, 0.0),
        ],
    )
    def test_omega(self, returns, risk_free, required_return, expected):
        assert_almost_equal(
            self.cofrics.omega_ratio(
                returns, risk_free=risk_free, required_return=required_return
            ),
            expected,
            DECIMAL_PLACES,
        )

    # As the required return increases (but is still less than the maximum
    # return), omega decreases
    @pytest.mark.parametrize(
        "returns, required_return_less, required_return_more",
        [
            (noise_uniform, 0.0, 0.001),
            (noise, 0.001, 0.002),
        ],
    )
    def test_omega_returns(self, returns, required_return_less, required_return_more):
        assert self.cofrics.omega_ratio(
            returns, required_return_less
        ) > self.cofrics.omega_ratio(returns, required_return_more)

    # Regressive Sharpe Ratio tests.
    @pytest.mark.parametrize(
        "returns, risk_free, expected",
        [
            (empty_returns, 0.0, np.nan),
            (one_return, 0.0, np.nan),
            (mixed_returns, mixed_returns, np.nan),
            (mixed_returns, 0.0, 1.7238613961706866),
            (mixed_returns, simple_benchmark, 0.34111411441060574),
            (positive_returns, 0.0, 52.915026221291804),
            (negative_returns, 0.0, -24.406808633910085),
            (flat_line_1, 0.0, np.inf),
        ],
    )
    def test_sharpe_ratio(self, returns, risk_free, expected):
        assert_almost_equal(
            self.cofrics.sharpe_ratio(returns, risk_free), expected, DECIMAL_PLACES
        )

    # Translate the returns and required returns by the same amount does not
    # change the Sharpe Ratio.
    @pytest.mark.parametrize(
        "returns, required_return, translation",
        [(noise_uniform, 0, 0.005), (noise_uniform, 0.005, 0.005)],
    )
    def test_sharpe_translation_same(self, returns, required_return, translation):
        sr = self.cofrics.sharpe_ratio(returns, required_return)
        sr_depressed = self.cofrics.sharpe_ratio(
            returns - translation, required_return - translation
        )
        sr_raised = self.cofrics.sharpe_ratio(
            returns + translation, required_return + translation
        )

        assert_almost_equal(sr, sr_depressed, DECIMAL_PLACES)
        assert_almost_equal(sr, sr_raised, DECIMAL_PLACES)

    # Translating the returns and required returns by the different amount
    # yields different Sharpe Ratio.
    @pytest.mark.parametrize(
        "returns, required_return, translation_returns, translation_required",
        [(noise_uniform, 0, 0.0002, 0.0001), (noise_uniform, 0.005, 0.0001, 0.0002)],
    )
    def test_sharpe_translation_diff(
        self, returns, required_return, translation_returns, translation_required
    ):
        sr = self.cofrics.sharpe_ratio(returns, required_return)
        sr_depressed = self.cofrics.sharpe_ratio(
            returns - translation_returns, required_return - translation_required
        )
        sr_raised = self.cofrics.sharpe_ratio(
            returns + translation_returns, required_return + translation_required
        )

        assert sr != sr_depressed
        assert sr != sr_raised

    # Translating the required return inversely affects the sharpe ratio.
    @pytest.mark.parametrize(
        "returns, required_return, translation",
        [(noise_uniform, 0.0, 0.005), (noise, 0.0, 0.005)],
    )
    def test_sharpe_translation_1(self, returns, required_return, translation):
        sr = self.cofrics.sharpe_ratio(returns, required_return)
        sr_depressed = self.cofrics.sharpe_ratio(returns, required_return - translation)
        sr_raised = self.cofrics.sharpe_ratio(returns, required_return + translation)

        assert sr_depressed > sr
        assert sr > sr_raised

    # Returns of a wider range or larger standard deviation decrease the
    # sharpe ratio.
    @pytest.mark.parametrize("small, large", [(0.001, 0.002), (0.01, 0.02)])
    def test_sharpe_noise(self, small, large):
        index = pd.date_range("2000-1-30", periods=1000)
        smaller_normal = pd.Series(rand.normal(0.01, small, 1000), index=index)
        larger_normal = pd.Series(rand.normal(0.01, large, 1000), index=index)

        assert self.cofrics.sharpe_ratio(
            smaller_normal, 0.001
        ) > self.cofrics.sharpe_ratio(larger_normal, 0.001)

    # Regressive downside risk tests.
    @pytest.mark.parametrize(
        "returns, required_return, period, expected",
        [
            (empty_returns, 0.0, DAILY, np.nan),
            (one_return, 0.0, DAILY, 0.0),
            (mixed_returns, mixed_returns, DAILY, 0.0),
            (mixed_returns, 0.0, DAILY, 0.60448325038829653),
            (mixed_returns, 0.1, DAILY, 1.7161730681956295),
            (weekly_returns, 0.0, WEEKLY, 0.25888650451930134),
            (weekly_returns, 0.1, WEEKLY, 0.7733045971672482),
            (monthly_returns, 0.0, MONTHLY, 0.1243650540411842),
            (monthly_returns, 0.1, MONTHLY, 0.37148351242013422),
            (
                df_simple,
                0.0,
                DAILY,
                pd.Series(
                    [0.20671788246185202, 0.083495680595704475], index=["one", "two"]
                ),
            ),
            (
                df_week,
                0.0,
                WEEKLY,
                pd.Series(
                    [0.093902996054410062, 0.037928477556776516], index=["one", "two"]
                ),
            ),
            (
                df_month,
                0.0,
                MONTHLY,
                pd.Series(
                    [0.045109540184877193, 0.018220251263412916], index=["one", "two"]
                ),
            ),
        ],
    )
    def test_downside_risk(self, returns, required_return, period, expected):
        downside_risk = self.cofrics.downside_risk(
            returns, required_return=required_return, period=period
        )

        # NOTE: using assert_almost_equal since np.nan != np.nan
        if isinstance(downside_risk, pd.Series):
            for i in range(downside_risk.size):
                assert_almost_equal(
                    downside_risk.iloc[i], expected.iloc[i], DECIMAL_PLACES
                )
        else:
            assert_almost_equal(downside_risk, expected, DECIMAL_PLACES)

    # As a higher percentage of returns are below the required return,
    # downside risk increases.
    @pytest.mark.parametrize(
        "noise, flat_line", [(noise, flat_line_0), (noise_uniform, flat_line_0)]
    )
    def test_downside_risk_noisy(self, noise, flat_line):
        noisy_returns_1 = noise[0:250].add(flat_line[250:], fill_value=0)
        noisy_returns_2 = noise[0:500].add(flat_line[500:], fill_value=0)
        noisy_returns_3 = noise[0:750].add(flat_line[750:], fill_value=0)

        dr_1 = self.cofrics.downside_risk(noisy_returns_1, flat_line)
        dr_2 = self.cofrics.downside_risk(noisy_returns_2, flat_line)
        dr_3 = self.cofrics.downside_risk(noisy_returns_3, flat_line)

        assert dr_1 <= dr_2
        assert dr_2 <= dr_3

    # Downside risk increases as the required_return increases
    @pytest.mark.parametrize(
        "returns, required_return", [(noise, 0.005), (noise_uniform, 0.005)]
    )
    def test_downside_risk_trans(self, returns, required_return):
        dr_0 = self.cofrics.downside_risk(returns, -required_return)
        dr_1 = self.cofrics.downside_risk(returns, 0)
        dr_2 = self.cofrics.downside_risk(returns, required_return)

        assert dr_0 <= dr_1
        assert dr_1 <= dr_2

    # Downside risk for a random series with a required return of 0 is higher
    # for datasets with larger standard deviation
    @pytest.mark.parametrize(
        "smaller_std, larger_std", [(0.001, 0.002), (0.001, 0.01), (0, 0.001)]
    )
    def test_downside_risk_std(self, smaller_std, larger_std):
        less_noise = pd.Series(
            rand.normal(0, smaller_std, 1000) if smaller_std != 0 else np.full(1000, 0),
            index=pd.date_range("2000-1-30", periods=1000, freq="D"),
        )
        more_noise = pd.Series(
            rand.normal(0, larger_std, 1000) if larger_std != 0 else np.full(1000, 0),
            index=pd.date_range("2000-1-30", periods=1000, freq="D"),
        )

        assert self.cofrics.downside_risk(less_noise) < self.cofrics.downside_risk(
            more_noise
        )

    # Regressive sortino ratio tests.
    @pytest.mark.parametrize(
        "returns, required_return, period, expected",
        [
            (empty_returns, 0.0, DAILY, np.nan),
            (one_return, 0.0, DAILY, np.nan),
            (mixed_returns, mixed_returns, DAILY, np.nan),
            (mixed_returns, 0.0, DAILY, 2.605531251673693),
            (mixed_returns, flat_line_1, DAILY, -1.3934779588919977),
            (positive_returns, 0.0, DAILY, np.inf),
            (negative_returns, 0.0, DAILY, -13.532743075043401),
            (simple_benchmark, 0.0, DAILY, np.inf),
            (weekly_returns, 0.0, WEEKLY, 1.1158901056866439),
            (monthly_returns, 0.0, MONTHLY, 0.53605626741889756),
            (
                df_simple,
                0.0,
                DAILY,
                pd.Series(
                    [3.0639640966566306, 38.090963117002495], index=["one", "two"]
                ),
            ),
            (
                df_week,
                0.0,
                WEEKLY,
                pd.Series(
                    [1.3918264112070571, 17.303077589064618], index=["one", "two"]
                ),
            ),
            (
                df_month,
                0.0,
                MONTHLY,
                pd.Series(
                    [0.6686117809312383, 8.3121296084492844], index=["one", "two"]
                ),
            ),
        ],
    )
    def test_sortino(self, returns, required_return, period, expected):
        sortino_ratio = self.cofrics.sortino_ratio(
            returns, required_return=required_return, period=period
        )

        if isinstance(sortino_ratio, pd.Series):
            for i in range(sortino_ratio.size):
                assert_almost_equal(
                    sortino_ratio.iloc[i], expected.iloc[i], DECIMAL_PLACES
                )
        else:
            assert_almost_equal(sortino_ratio, expected, DECIMAL_PLACES)

    # A large Sortino ratio indicates there is a low probability of a large
    # loss, therefore randomly changing values larger than required return to
    # a loss of 25 percent decreases the ratio.
    @pytest.mark.parametrize(
        "returns, required_return", [(noise_uniform, 0), (noise, 0)]
    )
    def test_sortino_add_noise(self, returns, required_return):
        # NOTE: don't mutate global test state.
        returns = returns.copy()
        sr_1 = self.cofrics.sortino_ratio(returns, required_return)
        upside_values = returns[returns > required_return].index.tolist()

        # Add large losses at random upside locations.
        loss_loc = rand.choice(upside_values, 2)
        returns[loss_loc[0]] = -0.01
        sr_2 = self.cofrics.sortino_ratio(returns, required_return)
        returns[loss_loc[1]] = -0.01
        sr_3 = self.cofrics.sortino_ratio(returns, required_return)

        assert sr_1 > sr_2
        assert sr_2 > sr_3

    # Similarly, randomly increasing some values below the required return to
    # the required return increases the ratio.
    @pytest.mark.parametrize(
        "returns, required_return", [(noise_uniform, 0), (noise, 0)]
    )
    def test_sortino_sub_noise(self, returns, required_return):
        # Don't mutate global test state
        returns = returns.copy()
        sr_1 = self.cofrics.sortino_ratio(returns, required_return)
        downside_values = returns[returns < required_return].index.tolist()
        # Replace some values below the required return to the required return
        loss_loc = rand.choice(downside_values, 2)
        returns[loss_loc[0]] = required_return
        sr_2 = self.cofrics.sortino_ratio(returns, required_return)
        returns[loss_loc[1]] = required_return
        sr_3 = self.cofrics.sortino_ratio(returns, required_return)

        assert sr_1 <= sr_2
        assert sr_2 <= sr_3

    # Translating the returns and required returns by the same amount
    # should not change the sortino ratio.
    @pytest.mark.parametrize(
        "returns, required_return, translation",
        [(noise_uniform, 0, 0.005), (noise_uniform, 0.005, 0.005)],
    )
    def test_sortino_translation_same(self, returns, required_return, translation):
        sr = self.cofrics.sortino_ratio(returns, required_return)
        sr_depressed = self.cofrics.sortino_ratio(
            returns - translation, required_return - translation
        )
        sr_raised = self.cofrics.sortino_ratio(
            returns + translation, required_return + translation
        )

        assert_almost_equal(sr, sr_depressed, DECIMAL_PLACES)
        assert_almost_equal(sr, sr_raised, DECIMAL_PLACES)

    # Translating the returns and required returns by the different amount
    # yields different sortino ratios.
    @pytest.mark.parametrize(
        "returns, required_return, translation_returns, translation_required",
        [(noise_uniform, 0, 0, 0.001), (noise_uniform, 0.005, 0.001, 0)],
    )
    def test_sortino_translation_diff(
        self, returns, required_return, translation_returns, translation_required
    ):
        sr = self.cofrics.sortino_ratio(returns, required_return)
        sr_depressed = self.cofrics.sortino_ratio(
            returns - translation_returns, required_return - translation_required
        )
        sr_raised = self.cofrics.sortino_ratio(
            returns + translation_returns, required_return + translation_required
        )

        assert sr != sr_depressed
        assert sr != sr_raised

    # Regressive tests for information ratio
    @pytest.mark.parametrize(
        "returns, factor_returns, expected",
        [
            (empty_returns, 0.0, np.nan),
            (one_return, 0.0, np.nan),
            (pos_line, pos_line, np.nan),
            (mixed_returns, 0.0, 0.10859306069076737),
            (mixed_returns, flat_line_1, -0.06515583641446039),
        ],
    )
    def test_excess_sharpe(self, returns, factor_returns, expected):
        assert_almost_equal(
            self.cofrics.excess_sharpe(returns, factor_returns),
            expected,
            DECIMAL_PLACES,
        )

    # The magnitude of the information ratio increases as a higher proportion
    # of returns are uncorrelated with the benchmark.
    @pytest.mark.parametrize(
        "noise_line, benchmark",
        [(flat_line_0, pos_line), (flat_line_1_tz, pos_line), (noise, pos_line)],
    )
    def test_excess_sharpe_noisy(self, noise_line, benchmark):
        noisy_returns_1 = noise_line[0:250].add(benchmark[250:], fill_value=0)
        noisy_returns_2 = noise_line[0:500].add(benchmark[500:], fill_value=0)
        noisy_returns_3 = noise_line[0:750].add(benchmark[750:], fill_value=0)

        ir_1 = self.cofrics.excess_sharpe(noisy_returns_1, benchmark)
        ir_2 = self.cofrics.excess_sharpe(noisy_returns_2, benchmark)
        ir_3 = self.cofrics.excess_sharpe(noisy_returns_3, benchmark)

        assert abs(ir_1) < abs(ir_2)
        assert abs(ir_2) < abs(ir_3)

    # Vertical translations change the information ratio in the direction of
    # the translation.
    @pytest.mark.parametrize(
        "returns, add_noise, translation",
        [
            (pos_line, noise, flat_line_1_tz),
            (pos_line, inv_noise, flat_line_1_tz),
            (neg_line, noise, flat_line_1_tz),
            (neg_line, inv_noise, flat_line_1_tz),
        ],
    )
    def test_excess_sharpe_trans(self, returns, add_noise, translation):
        ir = self.cofrics.excess_sharpe(returns + add_noise, returns)
        raised_ir = self.cofrics.excess_sharpe(
            returns + add_noise + translation, returns
        )
        depressed_ir = self.cofrics.excess_sharpe(
            returns + add_noise - translation, returns
        )

        assert ir < raised_ir
        assert depressed_ir < ir

    @pytest.mark.parametrize(
        "returns, benchmark, expected",
        [
            (empty_returns, simple_benchmark, (np.nan, np.nan)),
            (one_return, one_return, (np.nan, np.nan)),
            (
                mixed_returns,
                negative_returns[1:],
                (-0.9997853834885004, -0.71296296296296313),
            ),
            (mixed_returns, mixed_returns, (0.0, 1.0)),
            (mixed_returns, -mixed_returns, (0.0, -1.0)),
        ],
    )
    def test_alpha_beta(self, returns, benchmark, expected):
        alpha, beta = self.cofrics(
            pandas_only=len(returns) != len(benchmark),
            return_types=np.ndarray,
        ).alpha_beta(returns, benchmark)

        assert_almost_equal(alpha, expected[0], DECIMAL_PLACES)
        assert_almost_equal(beta, expected[1], DECIMAL_PLACES)

    # Regression tests for alpha.
    @pytest.mark.parametrize(
        "returns, benchmark, expected",
        [
            (empty_returns, simple_benchmark, np.nan),
            (one_return, one_return, np.nan),
            (mixed_returns, flat_line_1, np.nan),
            (mixed_returns, mixed_returns, 0.0),
            (mixed_returns, -mixed_returns, 0.0),
        ],
    )
    def test_alpha(self, returns, benchmark, expected):
        observed = self.cofrics.alpha(returns, benchmark)

        assert_almost_equal(observed, expected, DECIMAL_PLACES)

        if len(returns) == len(benchmark):
            # Compare to scipy.linregress
            returns_arr = returns.values
            benchmark_arr = benchmark.values
            mask = ~np.isnan(returns_arr) & ~np.isnan(benchmark_arr)
            try:
                slope, intercept, _, _, _ = stats.linregress(
                    benchmark_arr[mask], returns_arr[mask]
                )
            except ValueError:
                intercept = np.nan

            assert_almost_equal(observed, intercept * 252, DECIMAL_PLACES)

    # Alpha/beta translation tests.
    @pytest.mark.parametrize("mean_returns, translation", [(0, 0.001), (0.01, 0.001)])
    def test_alpha_beta_translation(self, mean_returns, translation):
        # Generate correlated returns and benchmark.
        std_returns, correlation, std_bench = 0.01, 0.8, 0.001
        means = [mean_returns, 0.001]
        covs = [
            [std_returns**2, std_returns * std_bench * correlation],
            [std_returns * std_bench * correlation, std_bench**2],
        ]
        (ret, bench) = rand.multivariate_normal(means, covs, 1000).T
        returns = pd.Series(
            data=ret, index=pd.date_range("2000-1-30", periods=1000, freq="D")
        )
        benchmark = pd.Series(
            bench, index=pd.date_range("2000-1-30", periods=1000, freq="D")
        )

        # Translate returns and generate alphas and betas.
        returns_depressed = returns - translation
        returns_raised = returns + translation
        alpha_beta = self.cofrics(return_types=np.ndarray).alpha_beta
        (alpha_depressed, beta_depressed) = alpha_beta(returns_depressed, benchmark)
        (alpha_standard, beta_standard) = alpha_beta(returns, benchmark)
        (alpha_raised, beta_raised) = alpha_beta(returns_raised, benchmark)

        # Alpha should change proportionally to how much return was translated.
        assert_almost_equal(
            ((alpha_standard + 1) ** (1 / 252)) - ((alpha_depressed + 1) ** (1 / 252)),
            translation,
            DECIMAL_PLACES,
        )
        assert_almost_equal(
            ((alpha_raised + 1) ** (1 / 252)) - ((alpha_standard + 1) ** (1 / 252)),
            translation,
            DECIMAL_PLACES,
        )
        # Beta remains constant.
        assert_almost_equal(beta_standard, beta_depressed, DECIMAL_PLACES)
        assert_almost_equal(beta_standard, beta_raised, DECIMAL_PLACES)

    # Test alpha/beta with a smaller and larger correlation values.
    @pytest.mark.parametrize("corr_less, corr_more", [(0.1, 0.9)])
    def test_alpha_beta_correlation(self, corr_less, corr_more):
        mean_returns = std_returns = 0.01
        mean_bench = std_bench = 0.001
        index = pd.date_range("2000-1-30", periods=1000)

        # Generate less correlated returns.
        means_less = [mean_returns, mean_bench]
        covs_less = [
            [std_returns**2, std_returns * std_bench * corr_less],
            [std_returns * std_bench * corr_less, std_bench**2],
        ]
        ret_less, bench_less = rand.multivariate_normal(means_less, covs_less, 1000).T
        returns_less = pd.Series(ret_less, index=index)
        benchmark_less = pd.Series(bench_less, index=index)

        # Generate more highly correlated returns.
        means_more = [mean_returns, mean_bench]
        covs_more = [
            [std_returns**2, std_returns * std_bench * corr_more],
            [std_returns * std_bench * corr_more, std_bench**2],
        ]
        (ret_more, bench_more) = rand.multivariate_normal(means_more, covs_more, 1000).T
        returns_more = pd.Series(ret_more, index=index)
        benchmark_more = pd.Series(bench_more, index=index)

        # Calculate alpha/beta values.
        alpha_beta = self.cofrics(return_types=np.ndarray).alpha_beta
        alpha_less, beta_less = alpha_beta(returns_less, benchmark_less)
        alpha_more, beta_more = alpha_beta(returns_more, benchmark_more)

        # Alpha determines by how much returns vary from the benchmark return.
        # A lower correlation leads to higher alpha.
        assert alpha_less > alpha_more

        # Beta measures the volatility of returns against benchmark returns.
        # Beta increase proportionally to correlation.
        assert beta_less < beta_more

    # When faced with data containing np.nan, do not return np.nan. Calculate
    # alpha and beta using dates containing both.
    @pytest.mark.parametrize("returns, benchmark", [(sparse_noise, sparse_noise)])
    def test_alpha_bta_with_nan_inputs(self, returns, benchmark):
        alpha, beta = self.cofrics(return_types=np.ndarray).alpha_beta(
            returns, benchmark
        )

        assert not np.isnan(alpha)
        assert not np.isnan(beta)

    @pytest.mark.parametrize(
        "returns, benchmark, expected, decimal_places",
        [
            (empty_returns, simple_benchmark, np.nan, None),
            (one_return, one_return, np.nan, None),
            (mixed_returns, flat_line_1, np.nan, None),
            (noise, noise, 1.0, None),
            (2 * noise, noise, 2.0, None),
            (noise, inv_noise, -1.0, None),
            (2 * noise, inv_noise, -2.0, None),
            (sparse_noise * flat_line_1_tz, sparse_flat_line_1_tz, np.nan, None),
            (
                simple_benchmark + rand.normal(0, 0.001, len(simple_benchmark)),
                pd.DataFrame({"returns": simple_benchmark}),
                1.0,
                2,
            ),
        ],
    )
    def test_beta(self, returns, benchmark, expected, decimal_places):
        decimal_places = DECIMAL_PLACES if decimal_places is None else decimal_places
        observed = self.cofrics.beta(returns, benchmark)
        assert_almost_equal(observed, expected, decimal_places)

        if len(returns) == len(benchmark):
            # Compare to scipy.linregress
            if isinstance(benchmark, pd.DataFrame):
                benchmark = benchmark["returns"]

            returns_arr = returns.values
            benchmark_arr = benchmark.values
            mask = ~np.isnan(returns_arr) & ~np.isnan(benchmark_arr)

            try:
                slope, intercept, _, _, _ = stats.linregress(
                    benchmark_arr[mask], returns_arr[mask]
                )
            except ValueError:
                slope = np.nan

            assert_almost_equal(observed, slope)

    @pytest.mark.parametrize(
        "returns, benchmark",
        [
            (empty_returns, simple_benchmark),
            (one_return, one_return),
            (mixed_returns, simple_benchmark[1:]),
            (mixed_returns, negative_returns[1:]),
            (mixed_returns, mixed_returns),
            (mixed_returns, -mixed_returns),
        ],
    )
    def test_alpha_beta_equality(self, returns, benchmark):
        alpha, beta = self.cofrics(
            pandas_only=len(returns) != len(benchmark),
            return_types=np.ndarray,
        ).alpha_beta(returns, benchmark)

        assert_almost_equal(
            alpha, self.cofrics.alpha(returns, benchmark), DECIMAL_PLACES
        )
        assert_almost_equal(beta, self.cofrics.beta(returns, benchmark), DECIMAL_PLACES)

        if len(returns) == len(benchmark):
            # Compare to scipy.linregress
            returns_arr = returns.values
            benchmark_arr = benchmark.values
            mask = ~np.isnan(returns_arr) & ~np.isnan(benchmark_arr)
            slope, intercept, _, _, _ = stats.linregress(
                returns_arr[mask], benchmark_arr[mask]
            )

            assert_almost_equal(alpha, intercept)
            assert_almost_equal(beta, slope)

    @pytest.mark.parametrize(
        "returns, expected",
        [
            (empty_returns, np.nan),
            (one_return, np.nan),
            (mixed_returns, 0.1529973665111273),
            (flat_line_1_tz, 1.0),
        ],
    )
    def test_stability_of_timeseries(self, returns, expected):
        assert_almost_equal(
            self.cofrics.stability_of_timeseries(returns), expected, DECIMAL_PLACES
        )

    @pytest.mark.parametrize(
        "returns, expected",
        [
            (empty_returns, np.nan),
            (one_return, 1.0),
            (mixed_returns, 0.9473684210526313),
            (pd.Series(rand.randn(100_000)), 1.0),
        ],
    )
    def test_tail_ratio(self, returns, expected):
        assert_almost_equal(self.cofrics.tail_ratio(returns), expected, 1)

    # Regression tests for Compound Annual Growth Rate (CAGR)
    @pytest.mark.parametrize(
        "returns, period, expected",
        [
            (empty_returns, DAILY, np.nan),
            (one_return, DAILY, 11.274002099240244),
            (mixed_returns, DAILY, 1.9135925373194231),
            (flat_line_1_tz, DAILY, 11.274002099240256),
            (
                pd.Series(
                    data=np.array([3.0, 3.0, 3.0]) / 100,
                    index=pd.date_range("2000-1-30", periods=3, freq="YE"),
                ),
                "yearly",
                0.03,
            ),
        ],
    )
    def test_cagr(self, returns, period, expected):
        assert_almost_equal(
            self.cofrics.cagr(returns, period=period), expected, DECIMAL_PLACES
        )

    # CAGR is calculated by the starting and ending value of returns.
    # Translating returns by a constant will change CAGR in the same direction
    # of translation.
    @pytest.mark.parametrize(
        "returns, constant", [(noise, 0.01), (noise_uniform, 0.01)]
    )
    def test_cagr_translation(self, returns, constant):
        cagr_depressed = self.cofrics.cagr(returns - constant)
        cagr_unchanged = self.cofrics.cagr(returns)
        cagr_raised = self.cofrics.cagr(returns + constant)

        assert cagr_depressed < cagr_unchanged
        assert cagr_unchanged < cagr_raised

    # Function does not return np.nan when inputs contains np.nan
    @pytest.mark.parametrize("returns", [sparse_noise])
    def test_cagr_with_nan_inputs(self, returns):
        assert not np.isnan(self.cofrics.cagr(returns))

    # Adding noise to returns should not significantly alter the cagr values.
    # Confirm that adding noise does not change cagr values to one
    # significant digit
    @pytest.mark.parametrize(
        "returns, add_noise",
        [(pos_line, noise), (pos_line, noise_uniform), (flat_line_1_tz, noise)],
    )
    def test_cagr_noisy(self, returns, add_noise):
        cagr = self.cofrics.cagr(returns)
        noisy_cagr_1 = self.cofrics.cagr(returns + add_noise)
        noisy_cagr_2 = self.cofrics.cagr(returns - add_noise)

        np.testing.assert_approx_equal(cagr, noisy_cagr_1, 1)
        np.testing.assert_approx_equal(cagr, noisy_cagr_2, 1)

    # Regression tests for beta_fragility_heuristic
    @pytest.mark.parametrize(
        "returns, factor_returns, expected",
        [
            (one_return, one_return, np.nan),
            (positive_returns, simple_benchmark, 0.0),
            (mixed_returns, simple_benchmark, 0.1),
            (negative_returns, simple_benchmark, -0.029999999999999999),
        ],
    )
    def test_beta_fragility_heuristic(self, returns, factor_returns, expected):
        assert_almost_equal(
            self.cofrics.beta_fragility_heuristic(returns, factor_returns),
            expected,
            DECIMAL_PLACES,
        )

    mixed_returns_expected_gpd_risk_result = [
        0.1,
        0.10001255835838491,
        1.5657360018514067e-06,
        0.4912526273742347,
        0.59126595492541179,
    ]

    negative_returns_expected_gpd_risk_result = [
        0.05,
        0.068353586736348199,
        9.4304947982121171e-07,
        0.34511639904932639,
        0.41347032855617882,
    ]

    # Regression tests for gpd_risk_estimates
    @pytest.mark.parametrize(
        "returns, expected",
        [
            (one_return, [0, 0, 0, 0, 0]),
            (empty_returns, [0, 0, 0, 0, 0]),
            (simple_benchmark, [0, 0, 0, 0, 0]),
            (positive_returns, [0, 0, 0, 0, 0]),
            (negative_returns, negative_returns_expected_gpd_risk_result),
            (mixed_returns, mixed_returns_expected_gpd_risk_result),
            (flat_line_1, [0, 0, 0, 0]),
            (weekly_returns, mixed_returns_expected_gpd_risk_result),
            (monthly_returns, mixed_returns_expected_gpd_risk_result),
        ],
    )
    def test_gpd_risk_estimates(self, returns, expected):
        result = self.cofrics.gpd_risk_estimates_aligned(returns)

        for result_item, expected_item in zip(result, expected):
            assert_almost_equal(result_item, expected_item, DECIMAL_PLACES)

    @pytest.mark.parametrize(
        "returns, window, expected",
        [
            (empty_returns, 6, []),
            (negative_returns, 6, [-0.2282, -0.2745, -0.2899, -0.2747]),
        ],
    )
    def test_roll_max_drawdown(self, returns, window, expected):
        test = self.cofrics.roll_max_drawdown(returns, window=window)

        assert_almost_equal(np.asanyarray(test), np.asanyarray(expected), 4)
        self.assert_indexes_match(test, returns[-len(expected) :])

    @pytest.mark.parametrize(
        "returns, window, expected",
        [
            (empty_returns, 6, []),
            (
                negative_returns,
                6,
                [-18.09162052, -26.79897486, -26.69138263, -25.72298838],
            ),
            (mixed_returns, 6, [7.57445259, 8.22784105, 8.22784105, -3.1374751]),
        ],
    )
    def test_roll_sharpe_ratio(self, returns, window, expected):
        test = self.cofrics.roll_sharpe_ratio(returns, window=window)

        assert_almost_equal(
            np.asanyarray(test), np.asanyarray(expected), DECIMAL_PLACES
        )
        self.assert_indexes_match(test, returns[-len(expected) :])

    @pytest.mark.parametrize(
        "returns, factor_returns, expected",
        [
            (empty_returns, empty_returns, np.nan),
            (one_return, one_return, 1.0),
            (mixed_returns, mixed_returns, 1.0),
            (all_negative_returns, mixed_returns, -0.52257643222960259),
        ],
    )
    def test_capture_ratio(self, returns, factor_returns, expected):
        test = self.cofrics.capture(returns, factor_returns)
        assert_almost_equal(test, expected, DECIMAL_PLACES)

    @pytest.mark.parametrize(
        "returns, factor_returns, expected",
        [
            (empty_returns, empty_returns, np.nan),
            (one_return, one_return, np.nan),
            (mixed_returns, mixed_returns, 1.0),
            (positive_returns, mixed_returns, -0.0006756053495),
            (all_negative_returns, mixed_returns, -0.0004338236),
        ],
    )
    def test_up_down_capture(self, returns, factor_returns, expected):
        test = self.cofrics.up_down_capture(returns, factor_returns)
        assert_almost_equal(test, expected, DECIMAL_PLACES)

    @pytest.mark.parametrize(
        "returns, factor_returns, expected",
        [
            (empty_returns, empty_returns, np.nan),
            (one_return, one_return, 1.0),
            (mixed_returns, mixed_returns, 1.0),
            (positive_returns, mixed_returns, 0.0076167762),
            (all_negative_returns, mixed_returns, -0.0004336328),
        ],
    )
    def test_up_capture(self, returns, factor_returns, expected):
        test = self.cofrics.up_capture(returns, factor_returns)
        assert_almost_equal(test, expected, DECIMAL_PLACES)

    @pytest.mark.parametrize(
        "returns, factor_returns, expected",
        [
            (empty_returns, empty_returns, np.nan),
            (one_return, one_return, np.nan),
            (mixed_returns, mixed_returns, 1.0),
            (all_negative_returns, mixed_returns, 0.99956025703798634),
            (positive_returns, mixed_returns, -11.27400221),
        ],
    )
    def test_down_capture(self, returns, factor_returns, expected):
        assert_almost_equal(
            self.cofrics.down_capture(returns, factor_returns), expected, DECIMAL_PLACES
        )

    @pytest.mark.parametrize(
        "returns, benchmark, window, expected",
        [
            (
                empty_returns,
                simple_benchmark,
                1,
                [(np.nan, np.nan)] * len(simple_benchmark),
            ),
            (one_return, one_return, 1, [(np.nan, np.nan)]),
            (
                mixed_returns,
                negative_returns,
                6,
                [
                    (-0.97854954, -0.7826087),
                    (-0.9828927, -0.76156584),
                    (-0.93166924, -0.61682243),
                    (-0.99967288, -0.41311475),
                ],
            ),
            (
                mixed_returns,
                mixed_returns,
                6,
                [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            ),
            (
                mixed_returns,
                -mixed_returns,
                6,
                [(0.0, -1.0), (0.0, -1.0), (0.0, -1.0), (0.0, -1.0)],
            ),
        ],
    )
    def test_roll_alpha_beta(self, returns, benchmark, window, expected):
        test = self.cofrics(return_types=(np.ndarray, pd.DataFrame)).roll_alpha_beta(
            returns, benchmark, window
        )

        if isinstance(test, pd.DataFrame):
            self.assert_indexes_match(test, benchmark[-len(expected) :])
            test = test.values

        alpha_test = [t[0] for t in test]
        beta_test = [t[1] for t in test]

        alpha_expected = [t[0] for t in expected]
        beta_expected = [t[1] for t in expected]

        assert_almost_equal(
            np.asarray(alpha_test), np.asarray(alpha_expected), DECIMAL_PLACES
        )
        assert_almost_equal(
            np.asarray(beta_test), np.asarray(beta_expected), DECIMAL_PLACES
        )

    @pytest.mark.parametrize(
        "returns, factor_returns, window, expected",
        [
            (empty_returns, empty_returns, 1, []),
            (one_return, one_return, 1, np.nan),
            (mixed_returns, mixed_returns, 6, [1.0, 1.0, 1.0, 1.0]),
            (
                positive_returns,
                mixed_returns,
                6,
                [-0.00011389, -0.00025861, -0.00015211, -0.00689239],
            ),
            (
                all_negative_returns,
                mixed_returns,
                6,
                [-6.38880246e-05, -1.65241701e-04, -1.65241719e-04, -6.89541957e-03],
            ),
        ],
    )
    def test_roll_up_down_capture(self, returns, factor_returns, window, expected):
        test = self.cofrics.roll_up_down_capture(returns, factor_returns, window=window)

        assert_almost_equal(np.asarray(test), np.asarray(expected), DECIMAL_PLACES)

    @pytest.mark.parametrize(
        "returns, factor_returns, window, expected",
        [
            (empty_returns, empty_returns, 1, []),
            (one_return, one_return, 1, [np.nan]),
            (mixed_returns, mixed_returns, 6, [1.0, 1.0, 1.0, 1.0]),
            (
                positive_returns,
                mixed_returns,
                6,
                [-11.2743862, -11.2743862, -11.2743862, -11.27400221],
            ),
            (
                all_negative_returns,
                mixed_returns,
                6,
                [0.92058591, 0.92058591, 0.92058591, 0.99956026],
            ),
        ],
    )
    def test_roll_down_capture(self, returns, factor_returns, window, expected):
        test = self.cofrics.roll_down_capture(returns, factor_returns, window=window)
        assert_almost_equal(np.asarray(test), np.asarray(expected), DECIMAL_PLACES)
        self.assert_indexes_match(test, returns[-len(expected) :])

    @pytest.mark.parametrize(
        "returns, factor_returns, window, expected",
        [
            (empty_returns, empty_returns, 1, []),
            (one_return, one_return, 1, [1.0]),
            (mixed_returns, mixed_returns, 6, [1.0, 1.0, 1.0, 1.0]),
            (
                positive_returns,
                mixed_returns,
                6,
                [0.00128406, 0.00291564, 0.00171499, 0.0777048],
            ),
            (
                all_negative_returns,
                mixed_returns,
                6,
                [-5.88144154e-05, -1.52119182e-04, -1.52119198e-04, -6.89238735e-03],
            ),
        ],
    )
    def test_roll_up_capture(self, returns, factor_returns, window, expected):
        test = self.cofrics.roll_up_capture(returns, factor_returns, window=window)
        assert_almost_equal(np.asarray(test), np.asarray(expected), DECIMAL_PLACES)
        self.assert_indexes_match(test, returns[-len(expected) :])

    @pytest.mark.parametrize(
        "returns, benchmark, expected",
        [
            (empty_returns, simple_benchmark, (np.nan, np.nan)),
            (one_return, one_return, (np.nan, np.nan)),
            (
                mixed_returns[1:],
                negative_returns[1:],
                (-0.9997853834885004, -0.71296296296296313),
            ),
            (mixed_returns, mixed_returns, (0.0, 1.0)),
            (mixed_returns, -mixed_returns, (0.0, -1.0)),
        ],
    )
    def test_down_alpha_beta(self, returns, benchmark, expected):
        down_alpha, down_beta = self.cofrics(
            pandas_only=len(returns) != len(benchmark),
            return_types=np.ndarray,
        ).down_alpha_beta(returns, benchmark)

        assert_almost_equal(down_alpha, expected[0], DECIMAL_PLACES)
        assert_almost_equal(down_beta, expected[1], DECIMAL_PLACES)

    @pytest.mark.parametrize(
        "returns, benchmark, expected",
        [
            (empty_returns, simple_benchmark, (np.nan, np.nan)),
            (one_return, one_return, (np.nan, np.nan)),
            (
                mixed_returns[1:],
                positive_returns[1:],
                (0.432961242076658, 0.4285714285),
            ),
            (mixed_returns, mixed_returns, (0.0, 1.0)),
            (mixed_returns, -mixed_returns, (0.0, -1.0)),
        ],
    )
    def test_up_alpha_beta(self, returns, benchmark, expected):
        up_alpha, up_beta = self.cofrics(
            pandas_only=len(returns) != len(benchmark),
            return_types=np.ndarray,
        ).up_alpha_beta(returns, benchmark)

        assert_almost_equal(up_alpha, expected[0], DECIMAL_PLACES)
        assert_almost_equal(up_beta, expected[1], DECIMAL_PLACES)

    def test_value_at_risk(self):
        value_at_risk = self.cofrics.value_at_risk
        returns = [1.0, 2.0]

        assert_almost_equal(value_at_risk(returns, cutoff=0.0), 1.0)
        assert_almost_equal(value_at_risk(returns, cutoff=0.3), 1.3)
        assert_almost_equal(value_at_risk(returns, cutoff=1.0), 2.0)

        returns = [1, 81, 82, 83, 84, 85]

        assert_almost_equal(value_at_risk(returns, cutoff=0.1), 41)
        assert_almost_equal(value_at_risk(returns, cutoff=0.2), 81)
        assert_almost_equal(value_at_risk(returns, cutoff=0.3), 81.5)

        # Test a returns stream of 21 data points at different cutoffs.
        returns = rand.normal(0, 0.02, 21)
        for cutoff in (0, 0.0499, 0.05, 0.20, 0.999, 1):
            assert_almost_equal(
                value_at_risk(returns, cutoff), np.percentile(returns, cutoff * 100)
            )

    def test_conditional_value_at_risk(self):
        value_at_risk = self.cofrics.value_at_risk
        conditional_value_at_risk = self.cofrics.conditional_value_at_risk

        # A single-valued array will always just have a CVaR of its only value.
        returns = rand.normal(0, 0.02, 1)
        expected_cvar = returns[0]

        assert_almost_equal(conditional_value_at_risk(returns, cutoff=0), expected_cvar)
        assert_almost_equal(conditional_value_at_risk(returns, cutoff=1), expected_cvar)

        # Test a returns stream of 21 data points at different cutoffs.
        returns = rand.normal(0, 0.02, 21)

        for cutoff in (0, 0.0499, 0.05, 0.20, 0.999, 1):
            # Find the VaR based on our cutoff, then take the average of all
            # values at or below the VaR.
            var = value_at_risk(returns, cutoff)
            expected_cvar = np.mean(returns[returns <= var])

            assert_almost_equal(
                conditional_value_at_risk(returns, cutoff), expected_cvar
            )

    @property
    def cofrics(self):
        """Return a wrapper around the cofrics module so tests can perform
        input conversions or return type checks on each call to a cofrics
        function.

        Each test case subclass can override this property, so that all the
        same tests are run, but with different function inputs or type checks.

        This was done as part of enabling cofrics functions to work with
        inputs of either pd.Series or np.ndarray, with the expectation that
        they will return the same type as their input.

        Returns
        -------
        cofrics

        Notes
        -----
        Since some parameterized test parameters refer to attributes on the
        real cofrics module at class body scope, this property must be
        defined later in the body than those references. That way, the
        attributes are looked up on the cofrics module, not this property.

        e.g. DAILY
        """
        return ReturnTypeCofricsProxy(self, (pd.Series, float))


class TestStatsArrays(TestStats):
    """Tests pass np.ndarray inputs to cofrics and assert that outputs are of
    type np.ndarray or float.
    """

    @property
    def cofrics(self):
        return PassArraysCofricsProxy(self, (np.ndarray, float))

    def assert_indexes_match(self, result, expected):
        pass


class TestStatsIntIndex(TestStats):
    """Tests pass int-indexed pd.Series inputs to cofrics and assert that outputs
    are of type pd.Series or float.

    This prevents a regression where we're indexing with ints into a pd.Series
    to find the last item and get KeyError when the series is int-indexed.

    """

    @property
    def cofrics(self):
        return ConvertPandasCofricsProxy(
            self,
            (pd.Series, float),
            lambda obj: type(obj)(obj.values, index=np.arange(len(obj))),
        )

    def assert_indexes_match(self, result, expected):
        pass


class TestHelpers(TestBaseCase):
    """Tests for helper methods and utils."""

    def setup_class(self):
        self.ser_length = 120
        self.window = 12
        self.returns = pd.Series(
            data=rand.randn(1, 120)[0] / 100.0,
            index=pd.date_range("2000-1-30", periods=120, freq="ME"),
        )
        self.factor_returns = pd.Series(
            rand.randn(1, 120)[0] / 100.0,
            index=pd.date_range("2000-1-30", periods=120, freq="ME"),
        )

    def test_roll_pandas(self):
        res = roll(self.returns, self.factor_returns, window=12, function=alpha_aligned)

        assert res.size == self.ser_length - self.window + 1

    def test_roll_ndarray(self):
        res = roll(
            self.returns.values,
            self.factor_returns.values,
            window=12,
            function=alpha_aligned,
        )

        assert len(res) == self.ser_length - self.window + 1

    def test_down(self):
        pd_res = down(self.returns, self.factor_returns, function=capture)
        np_res = down(self.returns.values, self.factor_returns.values, function=capture)

        assert isinstance(pd_res, float)
        assert_almost_equal(pd_res, np_res, DECIMAL_PLACES)

    def test_up(self):
        pd_res = up(self.returns, self.factor_returns, function=capture)
        np_res = up(self.returns.values, self.factor_returns.values, function=capture)

        assert isinstance(pd_res, float)
        assert_almost_equal(pd_res, np_res, DECIMAL_PLACES)

    def test_roll_bad_types(self):
        with pytest.raises(ValueError):
            roll(
                self.returns.values,
                self.factor_returns,
                window=12,
                function=max_drawdown,
            )

    def test_roll_max_window(self):
        res = roll(
            self.returns,
            self.factor_returns,
            window=self.ser_length + 100,
            function=max_drawdown,
        )
        assert res.size == 0


class Test2DStats(TestBaseCase):
    """Tests for functions that are capable of outputting a DataFrame."""

    input_one = [
        np.nan,
        0.01322056,
        0.03063862,
        -0.01422057,
        -0.00489779,
        0.01268925,
        -0.03357711,
        0.01797036,
    ]
    input_two = [
        0.01846232,
        0.00793951,
        -0.01448395,
        0.00422537,
        -0.00339611,
        0.03756813,
        0.0151531,
        np.nan,
    ]

    expected_0_one = [
        0.000000,
        0.013221,
        0.044264,
        0.029414,
        0.024372,
        0.037371,
        0.002539,
        0.020555,
    ]
    expected_0_two = [
        0.018462,
        0.026548,
        0.011680,
        0.015955,
        0.012504,
        0.050542,
        0.066461,
        0.066461,
    ]

    expected_100_one = [
        100.000000,
        101.322056,
        104.426424,
        102.941421,
        102.437235,
        103.737087,
        100.253895,
        102.055494,
    ]
    expected_100_two = [
        101.846232,
        102.654841,
        101.167994,
        101.595466,
        101.250436,
        105.054226,
        106.646123,
        106.646123,
    ]

    df_index = pd.date_range("2000-1-30", periods=8, freq="D")

    df_input = pd.DataFrame(
        {
            "one": pd.Series(input_one, index=df_index),
            "two": pd.Series(input_two, index=df_index),
        }
    )

    df_empty = pd.DataFrame()

    df_0_expected = pd.DataFrame(
        {
            "one": pd.Series(expected_0_one, index=df_index),
            "two": pd.Series(expected_0_two, index=df_index),
        }
    )

    df_100_expected = pd.DataFrame(
        {
            "one": pd.Series(expected_100_one, index=df_index),
            "two": pd.Series(expected_100_two, index=df_index),
        }
    )

    @pytest.mark.parametrize(
        "returns, starting_value, expected",
        [
            (df_input, 0, df_0_expected),
            (df_input, 100, df_100_expected),
            (df_empty, 0, pd.DataFrame()),
        ],
    )
    def test_cum_returns_df(self, returns, starting_value, expected):
        cum_returns = self.cofrics.cum_returns(returns, starting_value=starting_value)

        assert_almost_equal(np.asarray(cum_returns), np.asarray(expected), 4)
        self.assert_indexes_match(cum_returns, returns)

    @pytest.mark.parametrize(
        "returns, starting_value, expected",
        [
            (df_input, 0, df_0_expected.iloc[-1]),
            (df_input, 100, df_100_expected.iloc[-1]),
        ],
    )
    def test_cum_returns_final_df(self, returns, starting_value, expected):
        return_types = (pd.Series, np.ndarray)
        result = self.cofrics(return_types=return_types).cum_returns_final(
            returns, starting_value=starting_value
        )

        assert_almost_equal(np.array(result), expected, 5)
        self.assert_indexes_match(result, expected)

    @property
    def cofrics(self):
        """Returns a wrapper around the cofrics module so tests can perform
        input conversions or return type checks on each call to a cofrics
        function.

        See full explanation in TestStats.

        Returns
        -------
        cofrics

        """

        return ReturnTypeCofricsProxy(self, pd.DataFrame)


class Test2DStatsArrays(Test2DStats):
    """Tests pass np.ndarray inputs to cofrics and assert that outputs are of
    type np.ndarray.

    """

    @property
    def cofrics(self):
        return PassArraysCofricsProxy(self, np.ndarray)

    def assert_indexes_match(self, result, expected):
        pass


class ReturnTypeCofricsProxy:
    """A wrapper around the cofrics module which, on each function call, asserts
    that the type of the return value is in a given set.

    Also asserts that inputs were not modified by the cofrics function call.

    Calling an instance with kwargs will return a new copy with those
    attributes overridden.
    """

    def __init__(self, test_case, return_types):
        self._test_case = test_case
        self._return_types = return_types

    def __call__(self, **kwargs):
        dupe = copy(self)

        for k, v in kwargs.items():
            attr = "_" + k
            if hasattr(dupe, attr):
                setattr(dupe, attr, v)

        return dupe

    def __copy__(self):
        new_one = type(self).__new__(type(self))
        new_one.__dict__.update(self.__dict__)
        return new_one

    def __getattr__(self, item):
        return self._check_input_not_mutated(
            self._check_return_type(getattr(cofrics, item))
        )

    def _check_return_type(self, func):
        @wraps(func)
        def check_return_type(*args, **kwargs):
            result = func(*args, **kwargs)
            tuple_result = result if isinstance(result, tuple) else (result,)

            for r in tuple_result:
                assert isinstance(r, self._return_types)
            return result

        return check_return_type

    @staticmethod
    def _check_input_not_mutated(func):
        @wraps(func)
        def check_not_mutated(*args, **kwargs):
            # Copy inputs to compare them to originals later.
            arg_copies = [
                (i, arg.copy())
                for i, arg in enumerate(args)
                if isinstance(arg, (NDFrame, np.ndarray))
            ]
            kwarg_copies = {
                k: v.copy()
                for k, v in kwargs.items()
                if isinstance(v, (NDFrame, np.ndarray))
            }

            result = func(*args, **kwargs)

            # Check that inputs weren't mutated by func.
            for i, arg_copy in arg_copies:
                assert_allclose(
                    args[i],
                    arg_copy,
                    atol=0.5 * 10 ** (-DECIMAL_PLACES),
                    err_msg="Input 'arg %s' mutated by %s" % (i, func.__name__),
                )
            for kwarg_name, kwarg_copy in kwarg_copies.items():
                assert_allclose(
                    kwargs[kwarg_name],
                    kwarg_copy,
                    atol=0.5 * 10 ** (-DECIMAL_PLACES),
                    err_msg="Input '%s' mutated by %s" % (kwarg_name, func.__name__),
                )

            return result

        return check_not_mutated


class ConvertPandasCofricsProxy(ReturnTypeCofricsProxy):
    """A ReturnTypeCofricsProxy which also converts pandas NDFrame inputs to
    cofrics functions according to the given conversion method.

    Calling an instance with a truthy pandas_only will return a new instance
    which will skip the test when a cofrics function is called.

    """

    def __init__(self, test_case, return_types, convert, pandas_only=False):
        super(ConvertPandasCofricsProxy, self).__init__(test_case, return_types)
        self._convert = convert
        self._pandas_only = pandas_only

    def __getattr__(self, item):
        if self._pandas_only:
            raise pytest.skip(
                f"cofrics.{item} expects pandas-only inputs that have dt indexes/labels"
            )

        func = super(ConvertPandasCofricsProxy, self).__getattr__(item)

        @wraps(func)
        def convert_args(*args, **kwargs):
            args = [
                self._convert(arg) if isinstance(arg, NDFrame) else arg for arg in args
            ]
            kwargs = {
                k: self._convert(v) if isinstance(v, NDFrame) else v
                for k, v in kwargs.items()
            }
            return func(*args, **kwargs)

        return convert_args


class PassArraysCofricsProxy(ConvertPandasCofricsProxy):
    """A ConvertPandasCofricsProxy which converts NDFrame inputs to cofrics
    functions to numpy arrays.

    Calls the underlying
    cofrics.[alpha|beta|alpha_beta]_aligned functions directly, instead of
    the wrappers which align Series first.

    """

    def __init__(self, test_case, return_types):
        super().__init__(test_case, return_types, attrgetter("values"))

    def __getattr__(self, item):
        if item in (
            "alpha",
            "beta",
            "alpha_beta",
            "beta_fragility_heuristic",
            "gpd_risk_estimates",
        ):
            item += "_aligned"

        return super().__getattr__(item)
