"""Unit tests for the InputStore data container.

These tests exercise the store in isolation from Home Assistant. They use the
shared ``FakeStateMachine``/``FakeEntityState`` doubles from the root conftest
and a tiny in-memory storage double so values resolve identically to the config
loader (``resolve_field``/``resolve_constant``).
"""

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from conftest import FakeEntityState, FakeStateMachine
from custom_components.haeo.core.data.forecast_coverage import CoverageIssue, calculate_effective_horizon
from custom_components.haeo.core.data.input_store import InputMode, create_input_store
from custom_components.haeo.core.model.const import OutputType
from custom_components.haeo.core.schema import as_constant_value, as_entity_value
from custom_components.haeo.core.schema.field_hints import FieldHint

FORECAST_TIMESTAMPS: tuple[float, ...] = (0.0, 300.0, 600.0)


def _timestamps() -> tuple[float, ...]:
    """Return a fixed set of forecast timestamps for tests."""
    return FORECAST_TIMESTAMPS


class _MemStorage:
    """In-memory storage double implementing the Storage protocol."""

    def __init__(self, value: Any = None) -> None:
        self.value = value
        self.written: list[Any] = []

    def read(self) -> Any:
        return self.value

    async def write(self, value: Any) -> None:
        self.written.append(value)
        self.value = value


def _make_store(
    *,
    storage_value: Any = None,
    output_type: OutputType = OutputType.ENERGY,
    time_series: bool = False,
    boundaries: bool = False,
    negate: bool = False,
) -> Any:
    """Build an InputStore from an in-memory storage value and field hint."""
    storage = _MemStorage(storage_value)
    hint = FieldHint(output_type=output_type, time_series=time_series, boundaries=boundaries)
    return create_input_store(storage=storage, hint=hint, get_forecast_timestamps=_timestamps, negate=negate)


# --- Editable constant resolution ---


def test_editable_constant_scalar() -> None:
    """Editable scalar constant resolves to the value and is ready after refresh."""
    store = _make_store(storage_value=42.0, output_type=OutputType.ENERGY, time_series=False)

    assert store.mode == InputMode.EDITABLE
    assert store.source_entity_ids == []
    assert store.value == 42.0
    assert store.native_value == 42.0
    assert store.display_values == (42.0,)
    assert store.available is True

    # Resolved at construction but not marked ready until refresh/mark_ready.
    assert store.is_ready() is False
    store.refresh()
    assert store.is_ready() is True


def test_editable_constant_time_series_intervals() -> None:
    """Editable time-series interval constant expands to len(ts)-1 values."""
    store = _make_store(
        storage_value=as_constant_value(2.5),
        output_type=OutputType.POWER,
        time_series=True,
        boundaries=False,
    )

    assert isinstance(store.value, np.ndarray)
    assert len(store.value) == len(FORECAST_TIMESTAMPS) - 1
    np.testing.assert_array_equal(store.value, [2.5, 2.5])

    display = store.display_values
    assert display is not None
    assert len(display) == 2


def test_editable_constant_time_series_boundaries() -> None:
    """Editable boundary constant expands to len(ts) values."""
    store = _make_store(
        storage_value=10.0,
        output_type=OutputType.ENERGY,
        time_series=True,
        boundaries=True,
    )

    assert isinstance(store.value, np.ndarray)
    assert len(store.value) == len(FORECAST_TIMESTAMPS)
    np.testing.assert_array_equal(store.value, [10.0, 10.0, 10.0])


def test_editable_percent_field_divides_by_100() -> None:
    """Percent fields store the optimization value as a ratio but display as percent."""
    store = _make_store(storage_value=50.0, output_type=OutputType.STATE_OF_CHARGE, time_series=False)

    assert store.is_percent is True
    assert store.value == pytest.approx(0.5)
    assert store.native_value == pytest.approx(50.0)
    assert store.display_values == pytest.approx((50.0,))


def test_editable_bool_constant() -> None:
    """Boolean editable constants pass through unchanged."""
    store = _make_store(storage_value=as_constant_value(value=True), output_type=OutputType.ENERGY)

    assert store.mode == InputMode.EDITABLE
    assert store.value is True
    assert store.native_value is True
    assert store.display_values == (True,)


def test_hint_property_returns_field_hint() -> None:
    """The hint property exposes the store's field metadata."""
    store = _make_store(storage_value=1.0, output_type=OutputType.PRICE, time_series=True)

    assert store.hint.output_type == OutputType.PRICE
    assert store.hint.time_series is True


def test_display_values_none_when_not_loaded() -> None:
    """display_values is None before any value is resolved."""
    store = _make_store(output_type=OutputType.ENERGY, time_series=False)

    assert store.display_values is None


# --- Mutation: set_value / listeners / persist ---


def test_set_value_marks_ready_and_notifies() -> None:
    """set_value updates the value, marks ready, notifies listeners, and is available."""
    store = _make_store(output_type=OutputType.ENERGY, time_series=False)

    calls: list[int] = []
    store.add_listener(lambda: calls.append(1))

    assert store.is_ready() is False
    store.set_value(7.0)

    assert store.value == 7.0
    assert store.available is True
    assert store.is_ready() is True
    assert calls == [1]


def test_set_value_percent_divides() -> None:
    """set_value on a percent field divides the display value by 100."""
    store = _make_store(output_type=OutputType.STATE_OF_CHARGE, time_series=False)

    store.set_value(80.0)

    assert store.value == pytest.approx(0.8)
    assert store.native_value == pytest.approx(80.0)


def test_add_listener_unsub_removes_listener() -> None:
    """The unsubscribe callable removes the listener so it is no longer notified."""
    store = _make_store(output_type=OutputType.ENERGY, time_series=False)

    calls: list[int] = []
    unsub = store.add_listener(lambda: calls.append(1))

    store.set_value(1.0)
    assert calls == [1]

    unsub()
    store.set_value(2.0)
    assert calls == [1]


async def test_persist_writes_through_storage() -> None:
    """Persist forwards the schema value to the bound storage."""
    storage = _MemStorage()
    hint = FieldHint(output_type=OutputType.ENERGY)
    store = create_input_store(storage=storage, hint=hint, get_forecast_timestamps=_timestamps)

    await store.persist(as_constant_value(5.0))

    assert storage.written == [as_constant_value(5.0)]


# --- Driven async_load ---


async def test_driven_async_load_success() -> None:
    """Driven scalar load resolves from the source entity and becomes ready."""
    store = _make_store(
        storage_value=as_entity_value(["sensor.x"]),
        output_type=OutputType.ENERGY,
        time_series=False,
    )

    assert store.mode == InputMode.DRIVEN

    sm = FakeStateMachine({"sensor.x": FakeEntityState("sensor.x", "12.0", {})})
    assert await store.async_load(sm) is True

    assert store.value == 12.0
    assert store.available is True
    assert "sensor.x" in store.captured_source_states
    assert store.is_ready() is True


async def test_driven_async_load_negates_resolved_value() -> None:
    """A negated driven store flips the sign of values resolved from sources."""
    store = _make_store(
        storage_value=as_entity_value(["sensor.x"]),
        output_type=OutputType.PRICE,
        time_series=False,
        negate=True,
    )

    sm = FakeStateMachine({"sensor.x": FakeEntityState("sensor.x", "0.15", {})})
    assert await store.async_load(sm) is True

    assert store.value == pytest.approx(-0.15)
    assert store.native_value == pytest.approx(-0.15)


async def test_driven_async_load_negates_time_series() -> None:
    """Negation applies element-wise to time-series source values."""
    store = _make_store(
        storage_value=as_entity_value(["sensor.x"]),
        output_type=OutputType.PRICE,
        time_series=True,
        negate=True,
    )

    sm = FakeStateMachine(
        {"sensor.x": FakeEntityState("sensor.x", "0.2", {"forecast": [0.2, 0.3, 0.4]})},
    )
    assert await store.async_load(sm) is True

    assert isinstance(store.value, np.ndarray)
    assert np.all(store.value < 0)


@pytest.mark.parametrize(
    ("attributes", "issue"),
    [
        ({"forecast": []}, CoverageIssue.UNAVAILABLE),
        ({"forecast": "broken"}, CoverageIssue.UNSUPPORTED_STRUCTURE),
        (
            {"forecast": [{"time": "not-a-time", "value": 1.0}]},
            CoverageIssue.MALFORMED_TIMESTAMP,
        ),
        (
            {"forecast": [{"time": "2026-01-01T00:00:00+00:00", "value": "bad"}]},
            CoverageIssue.NON_NUMERIC_VALUE,
        ),
    ],
)
async def test_required_invalid_forecast_never_disappears_as_scalar(
    attributes: dict[str, object],
    issue: CoverageIssue,
) -> None:
    """A numeric fallback state cannot hide an invalid required forecast."""
    store = _make_store(
        storage_value=as_entity_value(["sensor.x"]),
        output_type=OutputType.POWER,
        time_series=True,
    )
    sm = FakeStateMachine({"sensor.x": FakeEntityState("sensor.x", "42", attributes)})
    assert await store.async_load(sm) is True

    required = store.required_forecasts
    assert len(required) == 1
    assert required[0].entity_id == "sensor.x"
    assert required[0].extraction_issue is issue


async def test_required_time_series_numeric_state_without_forecast_is_retained() -> None:
    """Configured time-series intent overrides a numeric scalar fallback."""
    store = _make_store(
        storage_value=as_entity_value(["sensor.x"]),
        output_type=OutputType.POWER,
        time_series=True,
    )
    sm = FakeStateMachine({"sensor.x": FakeEntityState("sensor.x", "42", {})})
    assert await store.async_load(sm) is True
    required = store.required_forecasts
    assert len(required) == 1
    assert required[0].entity_id == "sensor.x"
    assert required[0].extraction_issue is CoverageIssue.MISSING_FORECAST_ATTRIBUTE


async def test_resolve_for_horizon_preserves_irregular_explicit_final_boundary() -> None:
    """Strict store resolution uses the extracted N+1 boundary vector."""
    store = _make_store(
        storage_value=as_entity_value(["sensor.x"]),
        output_type=OutputType.POWER,
        time_series=True,
    )
    start = datetime(2026, 1, 1, tzinfo=UTC)
    boundaries = (
        start,
        start + timedelta(minutes=5),
        start + timedelta(minutes=10),
        start + timedelta(minutes=12, seconds=30),
    )
    state = FakeEntityState(
        "sensor.x",
        "0",
        {
            "unit_of_measurement": "kW",
            "forecast": [
                {"time": boundaries[0].isoformat(), "end_time": boundaries[1].isoformat(), "value": 1.0},
                {"time": boundaries[1].isoformat(), "end_time": boundaries[2].isoformat(), "value": 2.0},
                {"time": boundaries[2].isoformat(), "end_time": boundaries[3].isoformat(), "value": 3.0},
            ],
        },
    )
    assert await store.async_load(FakeStateMachine({"sensor.x": state})) is True
    epoch_boundaries = tuple(boundary.timestamp() for boundary in boundaries)

    assert store.resolve_for_horizon(epoch_boundaries) is True
    assert np.array_equal(store.value, np.array([1.0, 2.0, 3.0]))
    assert store.resolve_for_horizon((*epoch_boundaries[:-1], epoch_boundaries[-1] + 150.0)) is False


async def test_segmented_forecast_resolves_primary_then_extends_after_recovery() -> None:
    """An optional later segment extends the next run without stale coverage."""
    store = _make_store(
        storage_value=as_entity_value(["sensor.nordpool"]),
        output_type=OutputType.PRICE,
        time_series=True,
    )
    start = datetime(2026, 1, 1, tzinfo=UTC)
    middle = start + timedelta(hours=1)
    primary_end = start + timedelta(hours=2, minutes=15)
    later_end = start + timedelta(hours=3)

    def entry(begin: datetime, end: datetime, value: float) -> dict[str, object]:
        return {"start": begin.isoformat(), "end": end.isoformat(), "value": value}

    primary = [entry(start, middle, 1.0), entry(middle, primary_end, 2.0)]
    initial = FakeEntityState(
        "sensor.nordpool",
        "1.0",
        {"currency": "EUR", "raw_today": primary, "raw_tomorrow": None},
    )
    assert await store.async_load(FakeStateMachine({"sensor.nordpool": initial})) is True
    primary_horizon = (start.timestamp(), middle.timestamp(), primary_end.timestamp())
    assert store.resolve_for_horizon(primary_horizon) is True
    assert np.array_equal(store.value, np.array([1.0, 2.0]))
    assert store.resolve_for_horizon((*primary_horizon, later_end.timestamp())) is False

    recovered = FakeEntityState(
        "sensor.nordpool",
        "1.0",
        {
            "currency": "EUR",
            "raw_today": primary,
            "raw_tomorrow": [entry(primary_end, later_end, 3.0)],
        },
    )
    store.capture_state("sensor.nordpool", recovered)
    assert store.resolve_for_horizon((*primary_horizon, later_end.timestamp())) is True
    assert np.array_equal(store.value, np.array([1.0, 2.0, 3.0]))


async def test_nordpool_primary_resolves_when_horizon_starts_inside_interval() -> None:
    """Coverage and strict fusion agree for an intra-interval planning start."""
    store = _make_store(
        storage_value=as_entity_value(["sensor.nordpool"]),
        output_type=OutputType.PRICE,
        time_series=True,
    )
    start = datetime(2026, 1, 1, 12, tzinfo=UTC)
    boundaries = tuple(start + timedelta(hours=index) for index in range(4))
    state = FakeEntityState(
        "sensor.nordpool",
        "10",
        {
            "currency": "EUR",
            "raw_today": [
                {
                    "start": boundaries[index].isoformat(),
                    "end": boundaries[index + 1].isoformat(),
                    "value": value,
                }
                for index, value in enumerate((10.0, 20.0, 30.0))
            ],
            "raw_tomorrow": None,
        },
    )
    assert await store.async_load(FakeStateMachine({"sensor.nordpool": state})) is True
    horizon = (
        (start + timedelta(minutes=15)).timestamp(),
        (start + timedelta(minutes=30)).timestamp(),
        boundaries[1].timestamp(),
        boundaries[2].timestamp(),
    )
    coverage = calculate_effective_horizon(
        store.required_forecasts,
        configured_start=start + timedelta(minutes=15),
        configured_end=boundaries[2],
        required_timestamps=horizon,
    )
    assert coverage.effective_end == boundaries[2]
    assert store.resolve_for_horizon(horizon) is True
    assert np.array_equal(store.value, np.array([10.0, 10.0, 20.0]))

    beyond_source = (*horizon, boundaries[-1].timestamp() + 3600.0)
    assert store.resolve_for_horizon(beyond_source) is False


async def test_genuine_scalar_source_is_ignored_by_forecast_coverage() -> None:
    """A non-time-series input never participates in adaptive coverage."""
    store = _make_store(
        storage_value=as_entity_value(["sensor.x"]),
        output_type=OutputType.POWER,
        time_series=False,
    )
    sm = FakeStateMachine({"sensor.x": FakeEntityState("sensor.x", "unknown", {})})
    assert await store.async_load(sm) is False
    assert store.required_forecasts == ()


def test_negate_does_not_affect_editable_constant() -> None:
    """Constants are negated at the storage layer, so the store leaves them as-is."""
    store = _make_store(storage_value=as_constant_value(-0.15), output_type=OutputType.PRICE, negate=True)

    assert store.value == pytest.approx(-0.15)


def test_editable_none_storage_stays_unavailable() -> None:
    """An editable store with no configured value has no resolved data."""
    store = _make_store(storage_value={"type": "none"})

    store.refresh()

    assert store.value is None
    assert store.available is False
    assert store.display_values is None


async def test_driven_async_load_resolve_failure() -> None:
    """Resolution exceptions leave the store unavailable without clearing prior state."""
    store = _make_store(
        storage_value=as_entity_value(["sensor.x"]),
        output_type=OutputType.PRICE,
        time_series=False,
    )
    sm = FakeStateMachine({"sensor.x": FakeEntityState("sensor.x", "1.0", {})})

    with patch(
        "custom_components.haeo.core.data.input_store.resolve_field",
        side_effect=RuntimeError("boom"),
    ):
        assert await store.async_load(sm) is False

    assert store.available is False


async def test_driven_async_load_rejects_empty_time_series() -> None:
    """An empty time-series result is treated as a failed load."""
    store = _make_store(
        storage_value=as_entity_value(["sensor.x"]),
        output_type=OutputType.PRICE,
        time_series=True,
    )
    sm = FakeStateMachine({"sensor.x": FakeEntityState("sensor.x", "1.0", {})})

    with patch(
        "custom_components.haeo.core.data.input_store.resolve_field",
        return_value=np.array([]),
    ):
        assert await store.async_load(sm) is False

    assert store.available is False


async def test_driven_async_load_without_source_entities() -> None:
    """A driven store with no source entity IDs cannot load."""
    storage = _MemStorage({"type": "entity", "value": []})
    hint = FieldHint(output_type=OutputType.PRICE, time_series=False)
    store = create_input_store(storage=storage, hint=hint, get_forecast_timestamps=_timestamps)

    sm = FakeStateMachine({})
    assert await store.async_load(sm) is False
    assert store.available is False


async def test_driven_async_load_failure_keeps_unavailable() -> None:
    """A missing source entity leaves the store unavailable and not ready."""
    store = _make_store(
        storage_value=as_entity_value(["sensor.x"]),
        output_type=OutputType.ENERGY,
        time_series=False,
    )

    sm = FakeStateMachine({})
    assert await store.async_load(sm) is False

    assert store.available is False
    assert store.is_ready() is False
    assert store.value is None


# --- Construction errors ---


def test_create_input_store_invalid_config_raises() -> None:
    """An unrecognized schema dict raises a RuntimeError."""
    storage = _MemStorage({"type": "unknown_kind", "value": 1.0})
    hint = FieldHint(output_type=OutputType.ENERGY)

    with pytest.raises(RuntimeError, match="Invalid config value"):
        create_input_store(storage=storage, hint=hint, get_forecast_timestamps=_timestamps)


# --- horizon_start ---


def test_horizon_start_time_series_returns_first_timestamp() -> None:
    """Time-series editable stores expose the first forecast timestamp."""
    store = _make_store(storage_value=1.0, output_type=OutputType.POWER, time_series=True)

    store.refresh()

    assert store.horizon_start == FORECAST_TIMESTAMPS[0]


def test_horizon_start_scalar_returns_none() -> None:
    """Scalar stores have no loaded timestamps so horizon_start is None."""
    store = _make_store(storage_value=1.0, output_type=OutputType.ENERGY, time_series=False)

    store.refresh()

    assert store.horizon_start is None
