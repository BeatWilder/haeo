"""Tests for strict adaptive forecast coverage analysis."""

import ast
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from custom_components.haeo.core.data.forecast_coverage import (
    CoverageIssue,
    ForecastInput,
    calculate_effective_horizon,
    forecast_input_from_state,
    inspect_forecast,
)

START = datetime(2026, 1, 1, tzinfo=UTC)
END = START + timedelta(hours=12)
STEP = timedelta(hours=1)


def _series(hours: int, *, start: datetime = START) -> list[tuple[float, float]]:
    return [((start + index * STEP).timestamp(), float(index)) for index in range(hours)]


def test_all_inputs_cover_configured_horizon() -> None:
    """Full required inputs leave the configured horizon unchanged."""
    result = calculate_effective_horizon(
        [ForecastInput("sensor.pv", _series(13), boundary_values=True)],
        configured_start=START,
        configured_end=END,
    )
    assert result.effective_end == END
    assert result.coverage_ratio == 1.0
    assert not result.limited


def test_one_interval_input_ends_early() -> None:
    """N interval starts naturally cover through one cadence after the final start."""
    result = calculate_effective_horizon(
        [ForecastInput("sensor.load", _series(6))],
        configured_start=START,
        configured_end=END,
    )
    assert result.effective_end == START + timedelta(hours=6)
    assert result.reason is CoverageIssue.EARLY_END


def test_multiple_inputs_use_earliest_end() -> None:
    """The earliest required forecast limits the effective end."""
    result = calculate_effective_horizon(
        [
            ForecastInput("sensor.price", _series(10)),
            ForecastInput("sensor.pv", _series(8)),
            ForecastInput("sensor.load", _series(6)),
        ],
        configured_start=START,
        configured_end=END,
    )
    assert result.effective_end == START + timedelta(hours=6)
    assert result.limiting_input == "sensor.load"


def test_no_forecasts_represents_scalar_only_inputs() -> None:
    """Scalar/current-state inputs are excluded by callers and do not limit coverage."""
    result = calculate_effective_horizon([], configured_start=START, configured_end=END)
    assert result.effective_end == END
    assert result.inputs == ()


@pytest.mark.parametrize(
    ("forecast", "issue"),
    [
        (
            ForecastInput("sensor.duplicate", [(START.timestamp(), 1.0), (START.timestamp(), 2.0)]),
            CoverageIssue.DUPLICATE_TIMESTAMP,
        ),
        (
            ForecastInput(
                "sensor.non_monotonic",
                [
                    (START.timestamp(), 1.0),
                    ((START + STEP * 2).timestamp(), 2.0),
                    ((START + STEP).timestamp(), 3.0),
                ],
            ),
            CoverageIssue.NON_MONOTONIC_TIMESTAMP,
        ),
        (
            ForecastInput("sensor.malformed", [(float("nan"), 1.0), ((START + STEP).timestamp(), 2.0)]),
            CoverageIssue.MALFORMED_TIMESTAMP,
        ),
        (
            ForecastInput("sensor.non_numeric", [(START.timestamp(), 1.0), ((START + STEP).timestamp(), float("nan"))]),
            CoverageIssue.NON_NUMERIC_VALUE,
        ),
        (ForecastInput("sensor.unavailable", None), CoverageIssue.UNAVAILABLE),
    ],
)
def test_invalid_forecasts_fail_at_horizon_start(forecast: ForecastInput, issue: CoverageIssue) -> None:
    """Invalid required inputs have no safe usable horizon."""
    result = inspect_forecast(forecast, horizon_start=START, horizon_end=END)
    assert result.usable_end is None
    assert result.issue is issue


def test_stale_forecast() -> None:
    """A forecast ending at the start reports temporal coverage accurately."""
    stale = _series(3, start=START - timedelta(hours=3))
    result = inspect_forecast(ForecastInput("sensor.stale", stale), horizon_start=START, horizon_end=END)
    assert result.issue is CoverageIssue.DOES_NOT_COVER_START
    assert result.usable_end is None


def test_internal_gap_limits_continuous_coverage() -> None:
    """Coverage ends before a gap and never resumes after it."""
    series = _series(4)
    series.extend(_series(3, start=START + timedelta(hours=6)))
    result = inspect_forecast(ForecastInput("sensor.gapped", series), horizon_start=START, horizon_end=END)
    assert result.issue is CoverageIssue.INTERNAL_GAP
    assert result.usable_end == START + timedelta(hours=4)


def test_n_values_with_n_timestamps() -> None:
    """Interval-start values include the final interval's natural duration."""
    result = inspect_forecast(
        ForecastInput("sensor.intervals", _series(6), boundary_values=False),
        horizon_start=START,
        horizon_end=END,
    )
    assert result.usable_end == START + timedelta(hours=6)


def test_n_values_with_n_plus_one_boundary_timestamps() -> None:
    """Boundary values end exactly at their final timestamp."""
    result = inspect_forecast(
        ForecastInput("sensor.boundaries", _series(7), boundary_values=True),
        horizon_start=START,
        horizon_end=END,
    )
    assert result.usable_end == START + timedelta(hours=6)


def test_timezone_aware_requirement() -> None:
    """Naive horizon datetimes are rejected."""
    with pytest.raises(ValueError, match="timezone-aware"):
        calculate_effective_horizon(
            [],
            configured_start=datetime(2026, 1, 1),  # noqa: DTZ001 - intentionally invalid
            configured_end=datetime(2026, 1, 2),  # noqa: DTZ001 - intentionally invalid
        )


def test_daylight_saving_transition_uses_absolute_timestamps() -> None:
    """DST changes do not create false gaps in epoch-based coverage."""
    amsterdam = ZoneInfo("Europe/Amsterdam")
    local_start = datetime(2026, 3, 29, 0, 0, tzinfo=amsterdam)
    local_end = datetime(2026, 3, 29, 5, 0, tzinfo=amsterdam)
    absolute_start = local_start.astimezone(UTC)
    series = [((absolute_start + timedelta(hours=index)).timestamp(), float(index)) for index in range(5)]
    result = inspect_forecast(
        ForecastInput("sensor.dst", series, boundary_values=True),
        horizon_start=local_start,
        horizon_end=local_end,
    )
    assert result.issue is None
    assert result.usable_end == local_end


class _State:
    """Minimal Home Assistant state-shaped test double."""

    def __init__(self, entity_id: str, state: str, attributes: dict[str, object]) -> None:
        self.entity_id = entity_id
        self.state = state
        self.attributes = attributes


def test_scalar_state_is_ignored() -> None:
    """A scalar entity is not promoted to a required forecast."""
    state = _State("sensor.scalar", "42.0", {})
    source = forecast_input_from_state(
        state,
        boundary_values=False,
        expected_time_series=False,
    )
    assert source.scalar
    assert source.forecast is None


def test_unknown_scalar_state_is_ignored() -> None:
    """Configured scalar intent remains scalar even while unavailable."""
    state = _State("sensor.scalar", "unknown", {})
    source = forecast_input_from_state(
        state,
        boundary_values=False,
        expected_time_series=False,
    )
    assert source.scalar
    assert source.forecast is None


def test_required_numeric_state_without_forecast_attribute_is_missing() -> None:
    """Numeric current state cannot hide a required missing forecast."""
    state = _State("sensor.required", "42.0", {})
    source = forecast_input_from_state(
        state,
        boundary_values=False,
        expected_time_series=True,
    )
    assert not source.scalar
    assert source.forecast is not None
    assert source.forecast.extraction_issue is CoverageIssue.MISSING_FORECAST_ATTRIBUTE


def _nordpool_entry(start: datetime, end: datetime, value: object) -> dict[str, object]:
    return {"start": start.isoformat(), "end": end.isoformat(), "value": value}


@pytest.mark.parametrize("secondary", ["missing", None, []])
def test_segmented_forecast_keeps_valid_primary_when_later_segment_unavailable(
    secondary: object,
) -> None:
    """Missing, None, and empty optional later segments preserve raw_today."""
    irregular_end = START + timedelta(hours=2, minutes=15)
    attributes: dict[str, object] = {
        "raw_today": [
            _nordpool_entry(START, START + STEP, 1.0),
            _nordpool_entry(START + STEP, irregular_end, 2.0),
        ]
    }
    if secondary != "missing":
        attributes["raw_tomorrow"] = secondary
    source = forecast_input_from_state(
        _State("sensor.nordpool", "1.0", attributes),
        boundary_values=False,
        expected_time_series=True,
    )
    assert source.forecast is not None
    assert source.forecast.extraction_issue is None
    assert source.forecast.interval_boundaries == (
        START.timestamp(),
        (START + STEP).timestamp(),
        irregular_end.timestamp(),
    )


@pytest.mark.parametrize(
    ("secondary", "issue"),
    [
        ([{"start": "bad", "end": "bad", "value": 1.0}], CoverageIssue.MALFORMED_TIMESTAMP),
        (
            [_nordpool_entry(START + STEP, START + 2 * STEP, "bad")],
            CoverageIssue.NON_NUMERIC_VALUE,
        ),
        (["bad"], CoverageIssue.UNSUPPORTED_STRUCTURE),
        (
            [{"start": (START + STEP).isoformat(), "value": 1.0}],
            CoverageIssue.UNSUPPORTED_STRUCTURE,
        ),
    ],
)
def test_segmented_forecast_never_ignores_malformed_nonempty_later_segment(
    secondary: object,
    issue: CoverageIssue,
) -> None:
    """A present malformed later segment remains an explicit failure."""
    source = forecast_input_from_state(
        _State(
            "sensor.nordpool",
            "1.0",
            {
                "raw_today": [_nordpool_entry(START, START + STEP, 1.0)],
                "raw_tomorrow": secondary,
            },
        ),
        boundary_values=False,
        expected_time_series=True,
    )
    assert not source.scalar
    assert source.forecast is not None
    assert source.forecast.extraction_issue is issue


@pytest.mark.parametrize(
    "later_start",
    [START + timedelta(hours=1, minutes=15), START + timedelta(minutes=45)],
)
def test_segmented_forecast_rejects_gap_or_overlap(later_start: datetime) -> None:
    """Later segments must start exactly at the primary final boundary."""
    source = forecast_input_from_state(
        _State(
            "sensor.nordpool",
            "1.0",
            {
                "raw_today": [_nordpool_entry(START, START + STEP, 1.0)],
                "raw_tomorrow": [_nordpool_entry(later_start, later_start + STEP, 2.0)],
            },
        ),
        boundary_values=False,
        expected_time_series=True,
    )
    assert source.forecast is not None
    assert source.forecast.extraction_issue is CoverageIssue.INTERNAL_GAP


def test_malformed_raw_state_is_reported() -> None:
    """Malformed source timestamps remain visible to strict validation."""
    state = _State(
        "sensor.malformed",
        "unknown",
        {"forecast": [{"datetime": "not-a-timestamp", "value": 1.0}]},
    )
    source = forecast_input_from_state(
        state,
        boundary_values=False,
        expected_time_series=True,
    )
    assert source.forecast is not None
    result = inspect_forecast(source.forecast, horizon_start=START, horizon_end=END)
    assert result.issue is CoverageIssue.MALFORMED_TIMESTAMP


@pytest.mark.parametrize(
    ("state", "issue"),
    [
        (_State("sensor.empty", "42", {"forecast": []}), CoverageIssue.UNAVAILABLE),
        (
            _State("sensor.structure", "42", {"forecast": "broken"}),
            CoverageIssue.UNSUPPORTED_STRUCTURE,
        ),
        (
            _State(
                "sensor.value",
                "42",
                {"forecast": [{"time": "2026-01-01T00:00:00+00:00", "value": "bad"}]},
            ),
            CoverageIssue.NON_NUMERIC_VALUE,
        ),
        (
            _State("sensor.missing", "unknown", {}),
            CoverageIssue.UNAVAILABLE,
        ),
        (
            _State("sensor.missing_attribute", "not-a-number", {}),
            CoverageIssue.MISSING_FORECAST_ATTRIBUTE,
        ),
    ],
)
def test_invalid_forecast_source_never_becomes_scalar(
    state: _State,
    issue: CoverageIssue,
) -> None:
    """Known-invalid forecast sources remain explicit coverage failures."""
    source = forecast_input_from_state(
        state,
        boundary_values=False,
        expected_time_series=True,
    )
    assert not source.scalar
    assert source.forecast is not None
    assert source.forecast.extraction_issue is issue


def test_naive_source_timestamp_is_rejected() -> None:
    """A source wall-clock timestamp must include an explicit timezone."""
    state = _State(
        "sensor.naive",
        "0",
        {"forecast": [{"time": "2026-10-25T02:30:00", "value": 1.0}]},
    )
    source = forecast_input_from_state(
        state,
        boundary_values=False,
        expected_time_series=True,
    )
    assert source.forecast is not None
    assert source.forecast.extraction_issue is CoverageIssue.MALFORMED_TIMESTAMP


def test_one_interval_start_cannot_prove_an_end() -> None:
    """One start timestamp has no safe implicit final boundary."""
    result = inspect_forecast(
        ForecastInput("sensor.one", [(START.timestamp(), 1.0)]),
        horizon_start=START,
        horizon_end=END,
    )
    assert result.issue is CoverageIssue.AMBIGUOUS_CADENCE
    assert result.usable_end is None


def test_one_interval_with_explicit_end_is_valid() -> None:
    """One value plus two explicit boundaries uses the final boundary."""
    result = inspect_forecast(
        ForecastInput(
            "sensor.one_explicit",
            [(START.timestamp(), 1.0)],
            interval_boundaries=(
                START.timestamp(),
                (START + STEP).timestamp(),
            ),
        ),
        horizon_start=START,
        horizon_end=END,
    )
    assert result.usable_end == START + STEP
    assert result.issue is CoverageIssue.EARLY_END


def test_two_sparse_points_have_ambiguous_cadence() -> None:
    """Two starts cannot be stretched across an arbitrary large interval."""
    result = inspect_forecast(
        ForecastInput(
            "sensor.sparse",
            [
                (START.timestamp(), 1.0),
                ((START + timedelta(hours=8)).timestamp(), 2.0),
            ],
        ),
        horizon_start=START,
        horizon_end=END,
    )
    assert result.issue is CoverageIssue.AMBIGUOUS_CADENCE


def test_sustained_cadence_transition_is_supported() -> None:
    """Two unambiguous runs may transition to a new supported cadence."""
    timestamps = [
        START,
        START + STEP,
        START + STEP * 2,
        START + STEP * 4,
        START + STEP * 6,
    ]
    result = inspect_forecast(
        ForecastInput(
            "sensor.transition",
            [(timestamp.timestamp(), float(index)) for index, timestamp in enumerate(timestamps)],
        ),
        horizon_start=START,
        horizon_end=END,
    )
    assert result.usable_end == START + STEP * 8
    assert result.issue is CoverageIssue.EARLY_END


def test_irregular_isolated_interval_is_rejected() -> None:
    """A one-delta cadence regime is not treated as a valid transition."""
    timestamps = [
        START,
        START + STEP,
        START + STEP * 2,
        START + STEP * 5,
        START + STEP * 6,
        START + STEP * 7,
    ]
    result = inspect_forecast(
        ForecastInput(
            "sensor.irregular",
            [(timestamp.timestamp(), float(index)) for index, timestamp in enumerate(timestamps)],
        ),
        horizon_start=START,
        horizon_end=END,
    )
    assert result.issue is CoverageIssue.INTERNAL_GAP


def test_interval_boundary_length_mismatch_is_rejected() -> None:
    """N values require exactly N+1 explicit interval boundaries."""
    result = inspect_forecast(
        ForecastInput(
            "sensor.bad_lengths",
            _series(3),
            interval_boundaries=(START.timestamp(), (START + STEP).timestamp()),
        ),
        horizon_start=START,
        horizon_end=END,
    )
    assert result.issue is CoverageIssue.LENGTH_MISMATCH


def test_irregular_explicit_final_boundary_is_used_directly() -> None:
    """Explicit N+1 boundaries do not infer the final duration."""
    boundaries = (
        START.timestamp(),
        (START + STEP).timestamp(),
        (START + STEP * 2).timestamp(),
        (START + timedelta(hours=2, minutes=15)).timestamp(),
    )
    result = inspect_forecast(
        ForecastInput(
            "sensor.explicit",
            _series(3),
            interval_boundaries=boundaries,
        ),
        horizon_start=START,
        horizon_end=END,
    )
    assert result.usable_end == START + timedelta(hours=2, minutes=15)


def test_fall_back_folds_are_distinct_monotonic_utc_instants() -> None:
    """The repeated Amsterdam clock time remains ordered by UTC instant."""
    amsterdam = ZoneInfo("Europe/Amsterdam")
    first = datetime(2026, 10, 25, 2, 30, tzinfo=amsterdam, fold=0)
    second = datetime(2026, 10, 25, 2, 30, tzinfo=amsterdam, fold=1)
    state = _State(
        "sensor.fold",
        "0",
        {
            "forecast": [
                {"time": first.isoformat(), "value": 1.0},
                {"time": second.isoformat(), "value": 2.0},
                {"time": (second + timedelta(hours=1)).isoformat(), "value": 3.0},
            ]
        },
    )
    source = forecast_input_from_state(
        state,
        boundary_values=True,
        expected_time_series=True,
    )
    assert source.forecast is not None
    assert source.forecast.extraction_issue is None
    timestamps = [point[0] for point in source.forecast.series or ()]
    assert timestamps == sorted(timestamps)
    assert timestamps[0] != timestamps[1]


def test_two_sparse_boundary_values_do_not_prove_eight_hours() -> None:
    """Two boundary values only prove one exactly requested interval."""
    timestamps = [START.timestamp(), (START + timedelta(hours=8)).timestamp()]
    result = inspect_forecast(
        ForecastInput("sensor.sparse_boundary", [(timestamps[0], 1.0), (timestamps[1], 2.0)], boundary_values=True),
        horizon_start=START,
        horizon_end=START + timedelta(hours=8),
        required_timestamps=tuple((START + timedelta(hours=index)).timestamp() for index in range(9)),
    )
    assert result.issue is CoverageIssue.AMBIGUOUS_CADENCE
    assert result.usable_end is None


def test_two_boundary_values_cover_one_exact_interval() -> None:
    """Two points are valid when they exactly equal the one requested interval."""
    timestamps = (START.timestamp(), (START + STEP).timestamp())
    result = inspect_forecast(
        ForecastInput("sensor.single_boundary_interval", [(timestamps[0], 1.0), (timestamps[1], 2.0)], True),
        horizon_start=START,
        horizon_end=START + STEP,
        required_timestamps=timestamps,
    )
    assert result.issue is None
    assert result.usable_end == START + STEP


def test_boundary_values_missing_middle_boundary_stops_before_gap() -> None:
    """An isolated missing boundary limits coverage before the unsafe gap."""
    hours = (0, 1, 2, 4, 5, 6)
    result = inspect_forecast(
        ForecastInput(
            "sensor.boundary_gap",
            [((START + index * STEP).timestamp(), float(index)) for index in hours],
            boundary_values=True,
        ),
        horizon_start=START,
        horizon_end=START + timedelta(hours=6),
    )
    assert result.issue is CoverageIssue.INTERNAL_GAP
    assert result.usable_end == START + timedelta(hours=3)


def test_regular_boundary_values_cover_continuously() -> None:
    """A regular boundary sequence proves its complete range."""
    result = inspect_forecast(
        ForecastInput("sensor.regular_boundaries", _series(7), boundary_values=True),
        horizon_start=START,
        horizon_end=START + timedelta(hours=6),
    )
    assert result.issue is None
    assert result.usable_end == START + timedelta(hours=6)


def test_boundary_values_support_sustained_cadence_transition() -> None:
    """A cadence change is accepted only when both regimes are sustained."""
    hours = (0, 1, 2, 3, 5, 7, 9)
    result = inspect_forecast(
        ForecastInput(
            "sensor.boundary_transition",
            [((START + index * STEP).timestamp(), float(index)) for index in hours],
            boundary_values=True,
        ),
        horizon_start=START,
        horizon_end=START + timedelta(hours=9),
    )
    assert result.issue is None
    assert result.usable_end == START + timedelta(hours=9)


def test_boundary_values_reject_isolated_irregular_interval() -> None:
    """One isolated cadence cannot be treated as a sustained transition."""
    hours = (0, 1, 2, 4, 5, 6)
    result = inspect_forecast(
        ForecastInput(
            "sensor.boundary_irregular",
            [((START + index * STEP).timestamp(), float(index)) for index in hours],
            boundary_values=True,
        ),
        horizon_start=START,
        horizon_end=START + timedelta(hours=6),
    )
    assert result.issue is CoverageIssue.INTERNAL_GAP


def test_phase1a_domain_has_no_hardware_control_imports() -> None:
    """The new domain utility cannot acquire a hardware-control dependency."""
    source_path = Path(__file__).parents[1] / "forecast_coverage.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert imported_roots.isdisjoint({"aiohttp", "modbus", "mqtt", "pymodbus", "requests", "subprocess"})
    source = source_path.read_text(encoding="utf-8").lower()
    assert "services.async_call" not in source
    assert "dispatch" not in source
