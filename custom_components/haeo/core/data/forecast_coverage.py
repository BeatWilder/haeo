"""Pure forecast coverage analysis for adaptive optimization horizons."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from itertools import pairwise
import math
from typing import Final

from custom_components.haeo.core.state import EntityState

type ForecastSeries = Sequence[tuple[float, float]]

_CADENCE_REL_TOLERANCE: Final = 0.01
_MIN_INFERRED_CADENCE_DELTAS: Final = 2
_MIN_BOUNDARY_VALUES: Final = 2
_UNAVAILABLE_STATES: Final = frozenset({"unknown", "unavailable"})


class CoverageIssue(StrEnum):
    """Reason a forecast cannot cover more of the configured horizon."""

    EARLY_END = "early_ending_forecast"
    DUPLICATE_TIMESTAMP = "duplicate_timestamp"
    NON_MONOTONIC_TIMESTAMP = "non_monotonic_timestamp"
    MALFORMED_TIMESTAMP = "malformed_timestamp"
    INTERNAL_GAP = "internal_gap"
    UNAVAILABLE = "unavailable"
    NON_NUMERIC_VALUE = "non_numeric_value"
    UNSUPPORTED_STRUCTURE = "unsupported_forecast_structure"
    MISSING_FORECAST_ATTRIBUTE = "missing_forecast_attribute"
    DOES_NOT_COVER_START = "does_not_cover_start"
    AMBIGUOUS_CADENCE = "ambiguous_cadence"
    LENGTH_MISMATCH = "length_mismatch"
    INSUFFICIENT_COVERAGE = "insufficient_coverage"


class ForecastTimestampSemantics(StrEnum):
    """How source timestamps relate to interval values."""

    INTERVAL_STARTS = "interval_starts"
    INTERVAL_BOUNDARIES = "interval_boundaries"
    BOUNDARY_VALUES = "boundary_values"


@dataclass(frozen=True, slots=True)
class ForecastInput:
    """One required entity-backed time-series input."""

    entity_id: str
    series: ForecastSeries | None
    boundary_values: bool = False
    extraction_issue: CoverageIssue | None = None
    interval_boundaries: tuple[float, ...] | None = None
    detail: str | None = None
    present_value: float | None = None

    @property
    def semantics(self) -> ForecastTimestampSemantics:
        """Return the explicit timestamp/value convention."""
        if self.boundary_values:
            return ForecastTimestampSemantics.BOUNDARY_VALUES
        if self.interval_boundaries is not None:
            return ForecastTimestampSemantics.INTERVAL_BOUNDARIES
        return ForecastTimestampSemantics.INTERVAL_STARTS


@dataclass(frozen=True, slots=True)
class ForecastSource:
    """Typed classification of one configured entity source."""

    entity_id: str
    scalar: bool
    forecast: ForecastInput | None


@dataclass(frozen=True, slots=True)
class ForecastCoverage:
    """Continuous usable coverage for one forecast input."""

    entity_id: str
    usable_end: datetime | None
    issue: CoverageIssue | None
    issue_time: datetime | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class EffectiveHorizon:
    """Configured and effective horizon analysis."""

    configured_start: datetime
    configured_end: datetime
    effective_end: datetime
    limiting_input: str | None
    reason: CoverageIssue | None
    limiting_input_end: datetime | None
    coverage_ratio: float
    inputs: tuple[ForecastCoverage, ...]

    @property
    def limited(self) -> bool:
        """Return whether the effective end precedes the configured end."""
        return self.effective_end < self.configured_end


@dataclass(frozen=True, slots=True)
class _RawForecast:
    """Raw source-order values with optional explicit interval boundaries."""

    points: tuple[tuple[object, object], ...]
    interval_boundaries: tuple[object, ...] | None = None


def _aware_datetime(timestamp: float) -> datetime:
    """Convert a finite epoch timestamp to an aware UTC datetime."""
    if isinstance(timestamp, bool) or not math.isfinite(timestamp):
        raise ValueError
    return datetime.fromtimestamp(float(timestamp), tz=UTC)


def _parse_timestamp(raw_timestamp: object) -> float:
    """Parse one source timestamp, rejecting naive wall-clock values."""
    if isinstance(raw_timestamp, bool):
        raise TypeError
    if isinstance(raw_timestamp, (int, float)):
        return _aware_datetime(float(raw_timestamp)).timestamp()
    parsed = (
        raw_timestamp
        if isinstance(raw_timestamp, datetime)
        else datetime.fromisoformat(raw_timestamp)
        if isinstance(raw_timestamp, str)
        else None
    )
    if parsed is None or parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError
    return parsed.astimezone(UTC).timestamp()


def _parse_value(raw_value: object) -> float:
    """Parse a finite numeric forecast value."""
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float, str)):
        raise TypeError
    value = float(raw_value)
    if not math.isfinite(value):
        raise ValueError
    return value


def _same_cadence(left: float, right: float) -> bool:
    """Return whether two positive durations represent the same cadence."""
    return math.isclose(left, right, rel_tol=_CADENCE_REL_TOLERANCE, abs_tol=1e-6)


def _same_timestamp(left: float, right: float) -> bool:
    """Return whether two normalized epoch timestamps identify one instant."""
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-6)


def _cadence_runs(deltas: Sequence[float]) -> list[tuple[int, int, float]]:
    """Return inclusive-exclusive runs of approximately equal cadence."""
    if not deltas:
        return []
    runs: list[tuple[int, int, float]] = []
    start = 0
    cadence = deltas[0]
    for index, delta in enumerate(deltas[1:], start=1):
        if _same_cadence(delta, cadence):
            continue
        runs.append((start, index, cadence))
        start = index
        cadence = delta
    runs.append((start, len(deltas), cadence))
    return runs


def _validate_inferred_cadence(
    timestamps: Sequence[float],
    *,
    horizon_start: float,
) -> tuple[float | None, float | None, CoverageIssue | None]:
    """Validate interval starts and return final cadence or a safe gap end."""
    deltas = [current - previous for previous, current in pairwise(timestamps)]
    if any(delta <= 0 for delta in deltas):
        return None, None, CoverageIssue.NON_MONOTONIC_TIMESTAMP
    if len(deltas) < _MIN_INFERRED_CADENCE_DELTAS:
        return None, None, CoverageIssue.AMBIGUOUS_CADENCE

    runs = _cadence_runs(deltas)
    for run_index, (start, end, _cadence) in enumerate(runs):
        run_length = end - start
        if run_length >= _MIN_INFERRED_CADENCE_DELTAS:
            continue
        # A one-delta regime is ambiguous: it may be an isolated missing interval.
        issue = CoverageIssue.INTERNAL_GAP if 0 < run_index < len(runs) - 1 else CoverageIssue.AMBIGUOUS_CADENCE
        preceding_cadence = runs[run_index - 1][2] if run_index > 0 else 0.0
        gap_end = max(timestamps[start] + preceding_cadence, horizon_start)
        return None, gap_end, issue
    return deltas[-1], None, None


def inspect_forecast(
    forecast: ForecastInput,
    *,
    horizon_start: datetime,
    horizon_end: datetime,
    required_timestamps: Sequence[float] | None = None,
) -> ForecastCoverage:
    """Inspect continuous source coverage without interpolation or fabrication."""
    if horizon_start.tzinfo is None or horizon_end.tzinfo is None:
        msg = "Forecast horizon datetimes must be timezone-aware"
        raise ValueError(msg)
    start = horizon_start.astimezone(UTC)
    end = horizon_end.astimezone(UTC)
    if forecast.extraction_issue is not None:
        return ForecastCoverage(
            forecast.entity_id,
            None,
            forecast.extraction_issue,
            detail=forecast.detail,
        )
    if not forecast.series:
        return ForecastCoverage(forecast.entity_id, None, CoverageIssue.UNAVAILABLE)

    timestamps: list[float] = []
    for raw_timestamp, raw_value in forecast.series:
        try:
            timestamp = _aware_datetime(raw_timestamp).timestamp()
        except (OverflowError, OSError, ValueError):
            return ForecastCoverage(forecast.entity_id, None, CoverageIssue.MALFORMED_TIMESTAMP)
        if isinstance(raw_value, bool) or not math.isfinite(raw_value):
            return ForecastCoverage(forecast.entity_id, None, CoverageIssue.NON_NUMERIC_VALUE)
        timestamps.append(timestamp)

    if len(set(timestamps)) != len(timestamps):
        return ForecastCoverage(forecast.entity_id, None, CoverageIssue.DUPLICATE_TIMESTAMP)
    if any(current <= previous for previous, current in pairwise(timestamps)):
        return ForecastCoverage(forecast.entity_id, None, CoverageIssue.NON_MONOTONIC_TIMESTAMP)

    start_ts = start.timestamp()
    end_ts = end.timestamp()
    starts_after_horizon = timestamps[0] > start_ts
    leading_interval_is_seeded = (
        starts_after_horizon
        and forecast.present_value is not None
        and not forecast.boundary_values
        and forecast.interval_boundaries is None
        and required_timestamps is not None
        and len(required_timestamps) >= _MIN_BOUNDARY_VALUES
        and _same_timestamp(required_timestamps[0], start_ts)
        and _same_timestamp(required_timestamps[1], timestamps[0])
    )
    if (starts_after_horizon and not leading_interval_is_seeded) or timestamps[-1] < start_ts:
        return ForecastCoverage(forecast.entity_id, None, CoverageIssue.DOES_NOT_COVER_START)

    if forecast.boundary_values:
        if len(timestamps) < _MIN_BOUNDARY_VALUES:
            return ForecastCoverage(forecast.entity_id, None, CoverageIssue.AMBIGUOUS_CADENCE)
        if len(timestamps) == _MIN_BOUNDARY_VALUES:
            exact_single_interval = (
                required_timestamps is not None
                and len(required_timestamps) >= _MIN_BOUNDARY_VALUES
                and _same_timestamp(timestamps[0], required_timestamps[0])
                and _same_timestamp(timestamps[1], required_timestamps[1])
            )
            if not exact_single_interval:
                return ForecastCoverage(forecast.entity_id, None, CoverageIssue.AMBIGUOUS_CADENCE)
        else:
            _cadence, gap_end, issue = _validate_inferred_cadence(timestamps, horizon_start=start_ts)
            if issue is not None:
                usable_end = _aware_datetime(gap_end) if gap_end is not None and gap_end > start_ts else None
                return ForecastCoverage(forecast.entity_id, usable_end, issue)
        usable_end_ts = timestamps[-1]
    elif forecast.interval_boundaries is not None:
        boundaries = list(forecast.interval_boundaries)
        if len(boundaries) != len(timestamps) + 1:
            return ForecastCoverage(forecast.entity_id, None, CoverageIssue.LENGTH_MISMATCH)
        try:
            boundaries = [_aware_datetime(value).timestamp() for value in boundaries]
        except (OverflowError, OSError, ValueError):
            return ForecastCoverage(forecast.entity_id, None, CoverageIssue.MALFORMED_TIMESTAMP)
        if any(current <= previous for previous, current in pairwise(boundaries)):
            return ForecastCoverage(forecast.entity_id, None, CoverageIssue.NON_MONOTONIC_TIMESTAMP)
        if any(
            not _same_timestamp(timestamp, boundary)
            for timestamp, boundary in zip(
                timestamps,
                boundaries[:-1],
                strict=True,
            )
        ):
            return ForecastCoverage(forecast.entity_id, None, CoverageIssue.LENGTH_MISMATCH)
        usable_end_ts = boundaries[-1]
        if required_timestamps is not None:
            shared_ends = [
                timestamp
                for timestamp in required_timestamps
                if timestamp <= usable_end_ts and any(_same_timestamp(timestamp, boundary) for boundary in boundaries)
            ]
            usable_end_ts = max(shared_ends, default=start_ts)
    else:
        cadence, gap_end, issue = _validate_inferred_cadence(timestamps, horizon_start=start_ts)
        if issue is not None:
            usable_end = _aware_datetime(gap_end) if gap_end is not None and gap_end > start_ts else None
            return ForecastCoverage(forecast.entity_id, usable_end, issue)
        if cadence is None:
            return ForecastCoverage(forecast.entity_id, None, CoverageIssue.AMBIGUOUS_CADENCE)
        usable_end_ts = timestamps[-1] + cadence

    usable_end_ts = min(usable_end_ts, end_ts)
    usable_end = _aware_datetime(usable_end_ts)
    issue = CoverageIssue.EARLY_END if usable_end_ts < end_ts else None
    return ForecastCoverage(forecast.entity_id, usable_end, issue, usable_end if issue else None)


def calculate_effective_horizon(
    forecasts: Sequence[ForecastInput],
    *,
    configured_start: datetime,
    configured_end: datetime,
    required_timestamps: Sequence[float] | None = None,
) -> EffectiveHorizon:
    """Return the earliest continuous usable end across required forecasts."""
    if configured_start.tzinfo is None or configured_end.tzinfo is None:
        msg = "Forecast horizon datetimes must be timezone-aware"
        raise ValueError(msg)
    configured_start = configured_start.astimezone(UTC)
    configured_end = configured_end.astimezone(UTC)
    if configured_end <= configured_start:
        msg = "Configured horizon end must be after its start"
        raise ValueError(msg)

    inspected = tuple(
        inspect_forecast(
            source,
            horizon_start=configured_start,
            horizon_end=configured_end,
            required_timestamps=required_timestamps,
        )
        for source in forecasts
    )
    if not inspected:
        return EffectiveHorizon(
            configured_start,
            configured_end,
            configured_end,
            None,
            None,
            None,
            1.0,
            (),
        )

    limiting = min(inspected, key=lambda result: result.usable_end or configured_start)
    effective_end = limiting.usable_end or configured_start
    configured_seconds = (configured_end - configured_start).total_seconds()
    effective_seconds = max(0.0, (effective_end - configured_start).total_seconds())
    return EffectiveHorizon(
        configured_start=configured_start,
        configured_end=configured_end,
        effective_end=effective_end,
        limiting_input=limiting.entity_id,
        reason=limiting.issue,
        limiting_input_end=limiting.usable_end,
        coverage_ratio=min(1.0, effective_seconds / configured_seconds),
        inputs=inspected,
    )


def forecast_input_from_state(
    state: EntityState,
    *,
    boundary_values: bool,
    expected_time_series: bool,
) -> ForecastSource:
    """Classify an entity using the caller's configured source intent."""
    if not expected_time_series:
        return ForecastSource(entity_id=state.entity_id, scalar=True, forecast=None)

    raw = _find_raw_forecast(state)
    try:
        present_value = _parse_value(state.state)
    except (TypeError, ValueError):
        present_value = None
    if isinstance(raw, CoverageIssue):
        return ForecastSource(
            entity_id=state.entity_id,
            scalar=False,
            forecast=ForecastInput(state.entity_id, None, boundary_values, raw),
        )
    if isinstance(raw, _RawForecast):
        raw_series: list[tuple[float, float]] = []
        for raw_timestamp, raw_value in raw.points:
            try:
                timestamp = _parse_timestamp(raw_timestamp)
            except (OverflowError, OSError, TypeError, ValueError) as err:
                return ForecastSource(
                    entity_id=state.entity_id,
                    scalar=False,
                    forecast=ForecastInput(
                        state.entity_id,
                        None,
                        boundary_values,
                        CoverageIssue.MALFORMED_TIMESTAMP,
                        detail=str(err),
                    ),
                )
            try:
                value = _parse_value(raw_value)
            except (TypeError, ValueError) as err:
                return ForecastSource(
                    entity_id=state.entity_id,
                    scalar=False,
                    forecast=ForecastInput(
                        state.entity_id,
                        None,
                        boundary_values,
                        CoverageIssue.NON_NUMERIC_VALUE,
                        detail=str(err),
                    ),
                )
            raw_series.append((timestamp, value))
        try:
            interval_boundaries = (
                tuple(_parse_timestamp(timestamp) for timestamp in raw.interval_boundaries)
                if raw.interval_boundaries is not None
                else None
            )
        except (OverflowError, OSError, ValueError) as err:
            return ForecastSource(
                entity_id=state.entity_id,
                scalar=False,
                forecast=ForecastInput(
                    state.entity_id,
                    None,
                    boundary_values,
                    CoverageIssue.MALFORMED_TIMESTAMP,
                    detail=str(err),
                ),
            )
        return ForecastSource(
            entity_id=state.entity_id,
            scalar=False,
            forecast=ForecastInput(
                state.entity_id,
                tuple(raw_series),
                boundary_values,
                interval_boundaries=interval_boundaries,
                present_value=present_value,
            ),
        )

    if raw is None:
        issue = (
            CoverageIssue.UNAVAILABLE
            if state.state in _UNAVAILABLE_STATES
            else CoverageIssue.MISSING_FORECAST_ATTRIBUTE
        )
        return ForecastSource(
            entity_id=state.entity_id,
            scalar=False,
            forecast=ForecastInput(state.entity_id, None, boundary_values, issue),
        )

    return ForecastSource(
        entity_id=state.entity_id,
        scalar=False,
        forecast=ForecastInput(
            state.entity_id,
            None,
            boundary_values,
            CoverageIssue.UNSUPPORTED_STRUCTURE,
        ),
    )


def _find_raw_forecast(state: EntityState) -> _RawForecast | CoverageIssue | None:
    """Return source-order forecast data while preserving explicit boundaries."""
    timestamp_keys = ("time", "date", "period_start", "start_time", "start", "startsAt", "timestamp")
    end_keys = ("end_time", "end", "period_end")
    entity_name = state.entity_id.split(".", 1)[-1]
    known_sequence_attributes = (
        "forecast",
        "forecasts",
        "Forecasts",
        "detailedForecast",
        "prices",
        "raw_today",
        "deferrables_schedule",
        "predicted_temperatures",
        "battery_scheduled_power",
        "battery_scheduled_soc",
        "unit_load_cost_forecasts",
        "unit_prod_price_forecasts",
        "scheduled_forecast",
    )
    state_attributes: Mapping[str, object] = state.attributes
    raw_today = state.attributes.get("raw_today")
    raw_tomorrow = state.attributes.get("raw_tomorrow")
    if "raw_today" in state.attributes or "raw_tomorrow" in state.attributes:
        if not isinstance(raw_today, Sequence) or isinstance(raw_today, (str, bytes)):
            return CoverageIssue.UNSUPPORTED_STRUCTURE
        if raw_tomorrow is None:
            later_segment: Sequence[object] = ()
        elif isinstance(raw_tomorrow, Sequence) and not isinstance(raw_tomorrow, (str, bytes)):
            later_segment = raw_tomorrow
        else:
            return CoverageIssue.UNSUPPORTED_STRUCTURE
        state_attributes = {**state.attributes, "raw_today": [*raw_today, *later_segment]}

    for attribute in known_sequence_attributes:
        if attribute not in state_attributes:
            continue
        candidate = state_attributes[attribute]
        if not isinstance(candidate, Sequence) or isinstance(candidate, (str, bytes)):
            return CoverageIssue.UNSUPPORTED_STRUCTURE
        if not candidate:
            return CoverageIssue.UNAVAILABLE
        points: list[tuple[object, object]] = []
        starts: list[object] = []
        ends: list[object] = []
        for item in candidate:
            if not isinstance(item, Mapping):
                return CoverageIssue.UNSUPPORTED_STRUCTURE
            timestamp_key = next((key for key in timestamp_keys if key in item), None)
            if timestamp_key is None:
                return CoverageIssue.MALFORMED_TIMESTAMP
            value_key = next(
                (
                    key
                    for key in (
                        "value",
                        "power_w",
                        "pv_estimate",
                        "price",
                        "per_kwh",
                        "advanced_price_predicted",
                        entity_name,
                    )
                    if key in item
                ),
                None,
            )
            if value_key is None:
                return CoverageIssue.NON_NUMERIC_VALUE
            end_key = next((key for key in end_keys if key in item), None)
            starts.append(item[timestamp_key])
            if end_key is not None:
                ends.append(item[end_key])
            elif ends:
                return CoverageIssue.UNSUPPORTED_STRUCTURE
            points.append((item[timestamp_key], item[value_key]))
        boundaries: tuple[object, ...] | None = None
        if ends:
            if len(ends) != len(starts):
                return CoverageIssue.LENGTH_MISMATCH
            try:
                if any(
                    not _same_timestamp(
                        _parse_timestamp(end),
                        _parse_timestamp(start),
                    )
                    for end, start in zip(
                        ends[:-1],
                        starts[1:],
                        strict=True,
                    )
                ):
                    return CoverageIssue.INTERNAL_GAP
            except (OverflowError, OSError, TypeError, ValueError):
                return CoverageIssue.MALFORMED_TIMESTAMP
            boundaries = (*starts, ends[-1])
        return _RawForecast(tuple(points), boundaries)

    for attribute in ("forecast_dict", "watts"):
        if attribute not in state_attributes:
            continue
        candidate = state_attributes[attribute]
        if not isinstance(candidate, Mapping):
            return CoverageIssue.UNSUPPORTED_STRUCTURE
        if not candidate:
            return CoverageIssue.UNAVAILABLE
        return _RawForecast(tuple(candidate.items()))
    return None


__all__ = [
    "CoverageIssue",
    "EffectiveHorizon",
    "ForecastCoverage",
    "ForecastInput",
    "ForecastSource",
    "ForecastTimestampSemantics",
    "calculate_effective_horizon",
    "forecast_input_from_state",
    "inspect_forecast",
]
