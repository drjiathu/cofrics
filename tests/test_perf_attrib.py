import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from cofrics.perf_attrib import perf_attrib


class TestPerfAttrib:
    @staticmethod
    def test_perf_attrib_simple():
        start_date = "2017-01-01"
        periods = 2
        dts = pd.date_range(start_date, periods=periods, freq="B")
        dts.name = "dt"
        tickers = ["stock1", "stock2"]
        styles = ["risk_factor1", "risk_factor2"]
        returns = pd.Series(data=[0.1, 0.1], index=dts)
        factor_returns = pd.DataFrame(
            data={factor: [0.1, 0.1] for factor in styles}, columns=styles, index=dts
        )
        index = pd.MultiIndex.from_product([dts, tickers], names=["dt", "ticker"])

        positions = pd.Series(
            data=[
                0.2857142857142857,
                0.7142857142857143,
                0.2857142857142857,
                0.7142857142857143,
            ],
            index=index,
        )

        factor_loadings = pd.DataFrame(
            columns=styles,
            index=index,
            data={
                "risk_factor1": [0.25, 0.25, 0.25, 0.25],
                "risk_factor2": [0.25, 0.25, 0.25, 0.25],
            },
        )

        expected_perf_attrib_output = pd.DataFrame(
            index=dts,
            columns=[
                "risk_factor1",
                "risk_factor2",
                "total_returns",
                "common_returns",
                "specific_returns",
                "tilt_returns",
                "timing_returns",
            ],
            data={
                "risk_factor1": [0.025, 0.025],
                "risk_factor2": [0.025, 0.025],
                "common_returns": [0.05, 0.05],
                "specific_returns": [0.05, 0.05],
                "tilt_returns": [0.05, 0.05],
                "timing_returns": [0.0, 0.0],
                "total_returns": returns,
            },
        )

        expected_exposures_portfolio = pd.DataFrame(
            index=dts,
            columns=["risk_factor1", "risk_factor2"],
            data={"risk_factor1": [0.25, 0.25], "risk_factor2": [0.25, 0.25]},
        )

        exposures_portfolio, perf_attrib_output = perf_attrib(
            returns, positions, factor_returns, factor_loadings
        )

        # TODO: making consistence in `freq`.
        assert_frame_equal(
            expected_perf_attrib_output, perf_attrib_output, check_freq=False
        )
        assert_frame_equal(
            expected_exposures_portfolio, exposures_portfolio, check_freq=False
        )

        # test long and short positions
        positions = pd.Series([0.5, -0.5, 0.5, -0.5], index=index)
        exposures_portfolio, perf_attrib_output = perf_attrib(
            returns, positions, factor_returns, factor_loadings
        )

        expected_perf_attrib_output = pd.DataFrame(
            index=dts,
            columns=[
                "risk_factor1",
                "risk_factor2",
                "total_returns",
                "common_returns",
                "specific_returns",
                "tilt_returns",
                "timing_returns",
            ],
            data={
                "risk_factor1": [0.0, 0.0],
                "risk_factor2": [0.0, 0.0],
                "common_returns": [0.0, 0.0],
                "specific_returns": [0.1, 0.1],
                "tilt_returns": [0.0, 0.0],
                "timing_returns": [0.0, 0.0],
                "total_returns": returns,
            },
        )

        expected_exposures_portfolio = pd.DataFrame(
            index=dts,
            columns=["risk_factor1", "risk_factor2"],
            data={"risk_factor1": [0.0, 0.0], "risk_factor2": [0.0, 0.0]},
        )

        assert_frame_equal(
            expected_perf_attrib_output, perf_attrib_output, check_freq=False
        )
        assert_frame_equal(
            expected_exposures_portfolio, exposures_portfolio, check_freq=False
        )

        # test long and short positions with tilt exposure
        positions = pd.Series([1.0, -0.5, 1.0, -0.5], index=index)
        exposures_portfolio, perf_attrib_output = perf_attrib(
            returns, positions, factor_returns, factor_loadings
        )

        expected_perf_attrib_output = pd.DataFrame(
            index=dts,
            columns=[
                "risk_factor1",
                "risk_factor2",
                "total_returns",
                "common_returns",
                "specific_returns",
                "tilt_returns",
                "timing_returns",
            ],
            data={
                "risk_factor1": [0.0125, 0.0125],
                "risk_factor2": [0.0125, 0.0125],
                "common_returns": [0.025, 0.025],
                "specific_returns": [0.075, 0.075],
                "tilt_returns": [0.025, 0.025],
                "timing_returns": [0.0, 0.0],
                "total_returns": returns,
            },
        )

        expected_exposures_portfolio = pd.DataFrame(
            index=dts,
            columns=["risk_factor1", "risk_factor2"],
            data={"risk_factor1": [0.125, 0.125], "risk_factor2": [0.125, 0.125]},
        )

        assert_frame_equal(
            expected_perf_attrib_output, perf_attrib_output, check_freq=False
        )
        assert_frame_equal(
            expected_exposures_portfolio, exposures_portfolio, check_freq=False
        )

    def test_perf_attrib_regression(self):
        positions = pd.read_csv(
            "./tests/test_data/positions.csv", index_col=0, parse_dates=True
        )
        positions.columns = [
            int(col) if col != "cash" else col for col in positions.columns
        ]
        positions = positions.divide(positions.sum(axis="columns"), axis="rows")
        positions = positions.drop("cash", axis="columns").stack()

        returns = pd.read_csv(
            "./tests/test_data/returns.csv", index_col=0, parse_dates=True, header=None
        ).squeeze(axis="columns")
        factor_loadings = pd.read_csv(
            "./tests/test_data/factor_loadings.csv", index_col=[0, 1], parse_dates=[0]
        )
        factor_returns = pd.read_csv(
            "./tests/test_data/factor_returns.csv", index_col=0, parse_dates=True
        )
        residuals = pd.read_csv(
            "./tests/test_data/residuals.csv", index_col=0, parse_dates=True
        )
        residuals.columns = [int(col) for col in residuals.columns]
        intercepts = pd.read_csv(
            "./tests/test_data/intercepts.csv", index_col=0, header=None
        ).squeeze(axis="columns")
        risk_exposures_portfolio, perf_attrib_output = perf_attrib(
            returns, positions, factor_returns, factor_loadings
        )
        specific_returns = perf_attrib_output["specific_returns"]
        common_returns = perf_attrib_output["common_returns"]
        combined_returns = specific_returns + common_returns

        # Since all returns are factor returns, common returns should be
        # equivalent to total returns, and specific returns should be 0.
        assert_series_equal(returns, common_returns, check_names=False)
        assert np.isclose(specific_returns, 0).all()

        # Specific and common returns combined should equal total returns.
        assert_series_equal(returns, combined_returns, check_names=False)

        # Check that residuals + intercepts = specific returns
        assert np.isclose((residuals + intercepts), 0).all()

        # Check that exposure * factor returns = common returns
        expected_common_returns = risk_exposures_portfolio.multiply(
            factor_returns, axis="rows"
        ).sum(axis="columns")
        assert_series_equal(expected_common_returns, common_returns, check_names=False)

        # Since factor loadings are ones, portfolio risk exposures should be the same.
        assert_frame_equal(
            risk_exposures_portfolio,
            pd.DataFrame(
                data=np.ones_like(risk_exposures_portfolio),
                index=risk_exposures_portfolio.index,
                columns=risk_exposures_portfolio.columns,
            ),
        )
