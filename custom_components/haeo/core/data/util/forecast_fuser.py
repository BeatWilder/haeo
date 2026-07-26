"""Fuse combined forecast data into horizon-aligned values."""

from collections.abc import Sequence
from itertools import pairwise
from typing import Any

import numpy as np
from numpy.typing import NDArray

from custom_components.haeo.core.data.util.forecast_cycle import normalize_forecast_cycle

from . import ForecastSeries

# Need at least 2 boundaries (start and end) to define one interval
MIN_BOUNDARIES = 2


def _build_extended_block(
    forecast_series: ForecastSeries,
    horizon_start: float,
    horizon_end: float,
) -> NDArray[Any]:
    """Build extended forecast block covering the horizon with cycling.

    Args:
        forecast_series: Time series forecast data (must not be empty)
        horizon_start: Start of the horizon
        horizon_end: End of the horizon

    Returns:
        Structured numpy array with 'timestamp' and 'value' fields

    """
    block, cover_seconds = normalize_forecast_cycle(forecast_series, horizon_start)

    # Repeat block as needed to cover the entire horizon
    repeat_count = max(2, int(np.ceil((horizon_end - horizon_start) / cover_seconds)) + 1)
    extended = [(timestamp + i * cover_seconds, value) for i in range(repeat_count) for (timestamp, value) in block]
    return np.array(extended, dtype=[("timestamp", np.float64), ("value", np.float64)])


def _build_strict_block(
    forecast_series: ForecastSeries,
    horizon_start: float,
    horizon_end: float,
    *,
    interval_starts: bool = False,
    allow_single_interval: bool = False,
    allow_leading_present_interval: bool = False,
) -> NDArray[Any]:
    """Build a source block only when it naturally covers the horizon."""
    block = np.array(forecast_series, dtype=[("timestamp", np.float64), ("value", np.float64)])
    if block.size == 0 or (block[0]["timestamp"] > horizon_start and not allow_leading_present_interval):
        msg = "Forecast does not continuously cover the requested horizon"
        raise ValueError(msg)
    if block.size == 1:
        if not interval_starts or not allow_single_interval or block[0]["timestamp"] != horizon_start:
            msg = "Forecast does not continuously cover the requested horizon"
            raise ValueError(msg)
        # Adaptive coverage only permits a one-point interval series when its
        # explicit final boundary has already limited ``horizon_end``.
        return np.append(
            block,
            np.array([(horizon_end, block[0]["value"])], dtype=block.dtype),
        )
    if interval_starts and block[-1]["timestamp"] < horizon_end:
        # Coverage validation has already established that the final local
        # cadence is unambiguous. Do not use a broad/global median here because
        # a supported sustained resolution transition may precede the final run.
        cadence = float(block[-1]["timestamp"] - block[-2]["timestamp"])
        natural_end = float(block[-1]["timestamp"]) + cadence
        if cadence <= 0 or natural_end < horizon_end:
            msg = "Forecast does not continuously cover the requested horizon"
            raise ValueError(msg)
        block = np.append(block, np.array([(natural_end, block[-1]["value"])], dtype=block.dtype))
    elif block[-1]["timestamp"] < horizon_end:
        msg = "Forecast does not continuously cover the requested horizon"
        raise ValueError(msg)
    return block


def fuse_to_boundaries(
    present_value: float | None,
    forecast_series: ForecastSeries,
    horizon_times: Sequence[float],
) -> list[float]:
    """Fuse a combined forecast into point-in-time values at each horizon boundary.

    Args:
        present_value: Current sensor value (actual current state)
        forecast_series: Time series forecast data
        horizon_times: Boundary timestamps (n+1 values defining n intervals)

    Returns:
        n+1 point-in-time values where:
        - Position 0: Present value at horizon_times[0] (actual current state if provided)
        - Position k (k≥1): Interpolated value at horizon_times[k]

    """
    if not horizon_times:
        return []

    # Can't make any values if both forecast and present_value are missing
    if not forecast_series and present_value is None:
        msg = "Either forecast_series or present_value must be provided."
        raise ValueError(msg)

    # Just a present value, no forecast - return it for all boundaries
    if not forecast_series and present_value is not None:
        return [present_value] * len(horizon_times)

    block_array = _build_extended_block(forecast_series, horizon_times[0], horizon_times[-1])

    # Interpolate at boundary times
    values = np.interp(horizon_times, block_array["timestamp"], block_array["value"])

    # Replace position 0 with present_value if provided
    result = [float(v) for v in values]
    if present_value is not None:
        result[0] = present_value
    return result


def fuse_to_boundaries_strict(
    present_value: float | None,
    forecast_series: ForecastSeries,
    horizon_times: Sequence[float],
) -> list[float]:
    """Fuse boundary values without cycling or extrapolating source data."""
    if not horizon_times:
        return []
    if not forecast_series:
        msg = "A forecast series is required for strict horizon fusion"
        raise ValueError(msg)
    block = _build_strict_block(forecast_series, horizon_times[0], horizon_times[-1])
    result = [float(value) for value in np.interp(horizon_times, block["timestamp"], block["value"])]
    if present_value is not None:
        result[0] = present_value
    return result


def fuse_to_intervals(
    present_value: float | None,
    forecast_series: ForecastSeries,
    horizon_times: Sequence[float],
) -> list[float]:
    """Fuse a combined forecast into interval averages aligned with the horizon.

    Args:
        present_value: Current sensor value (actual current state)
        forecast_series: Time series forecast data
        horizon_times: Boundary timestamps (n+1 values defining n intervals)

    Returns:
        n interval values where:
        - Position 0: Present value (actual current state) if provided, else trapezoidal average
        - Position k (k≥1): Trapezoidal average over interval [horizon_times[k], horizon_times[k+1]]

    Trapezoidal integration accounts for internal forecast points within each interval,
    not just the endpoint values.

    """
    if not horizon_times or len(horizon_times) < MIN_BOUNDARIES:
        return []

    n_intervals = len(horizon_times) - 1

    # No forecast: broadcast present value to all intervals
    if not forecast_series:
        if present_value is None:
            msg = "Either forecast_series or present_value must be provided."
            raise ValueError(msg)
        return [present_value] * n_intervals

    horizon_start = horizon_times[0]
    horizon_end = horizon_times[-1]

    block_array = _build_extended_block(forecast_series, horizon_start, horizon_end)

    # Trapezoidal integration over each interval
    result: list[float] = []
    for i in range(n_intervals):
        interval_start = horizon_times[i]
        interval_end = horizon_times[i + 1]
        interval_duration = interval_end - interval_start

        # Get block points strictly within this interval (excluding boundaries)
        mask = (block_array["timestamp"] > interval_start) & (block_array["timestamp"] < interval_end)
        interval_points = block_array[mask]

        # Build integration series: start boundary + internal points + end boundary
        start_value = np.interp(interval_start, block_array["timestamp"], block_array["value"])
        end_value = np.interp(interval_end, block_array["timestamp"], block_array["value"])
        times = np.concatenate([[interval_start], interval_points["timestamp"], [interval_end]])
        values = np.concatenate([[start_value], interval_points["value"], [end_value]])

        # Trapezoidal integration: area under curve divided by duration
        area = np.trapezoid(values, times)
        result.append(float(area / interval_duration))

    # Replace first interval with present_value if provided
    if present_value is not None:
        result[0] = present_value

    return result


def fuse_to_intervals_strict(
    present_value: float | None,
    forecast_series: ForecastSeries,
    horizon_times: Sequence[float],
    *,
    allow_single_interval: bool = False,
    interval_boundaries: Sequence[float] | None = None,
) -> list[float]:
    """Fuse interval averages without cycling or extrapolating source data."""
    if not horizon_times or len(horizon_times) < MIN_BOUNDARIES:
        return []
    if not forecast_series:
        msg = "A forecast series is required for strict horizon fusion"
        raise ValueError(msg)

    if interval_boundaries is not None:
        boundaries = [float(timestamp) for timestamp in interval_boundaries]
        if len(boundaries) != len(forecast_series) + 1:
            msg = "Explicit interval boundaries must contain N+1 timestamps for N values"
            raise ValueError(msg)
        starts = [float(timestamp) for timestamp, _value in forecast_series]
        if any(current <= previous for previous, current in pairwise(boundaries)) or any(
            not np.isclose(start, boundary, rtol=0.0, atol=1e-6)
            for start, boundary in zip(starts, boundaries[:-1], strict=True)
        ):
            msg = "Explicit interval boundaries do not match forecast interval starts"
            raise ValueError(msg)
        if (
            horizon_times[0] < boundaries[0]
            or horizon_times[0] >= boundaries[-1]
            or not any(np.isclose(horizon_times[-1], boundary, rtol=0.0, atol=1e-6) for boundary in boundaries)
            or horizon_times[-1] > boundaries[-1]
        ):
            msg = "Requested horizon must start within coverage and end on an explicit interval boundary"
            raise ValueError(msg)

        source_values = [float(value) for _timestamp, value in forecast_series]
        result: list[float] = []
        for interval_start, interval_end in pairwise(horizon_times):
            duration = interval_end - interval_start
            if duration <= 0:
                msg = "Requested horizon boundaries must be strictly increasing"
                raise ValueError(msg)
            weighted_total = 0.0
            covered = 0.0
            for source_start, source_end, source_value in zip(
                boundaries[:-1],
                boundaries[1:],
                source_values,
                strict=True,
            ):
                overlap = max(0.0, min(interval_end, source_end) - max(interval_start, source_start))
                weighted_total += overlap * source_value
                covered += overlap
            if not np.isclose(covered, duration, rtol=0.0, atol=1e-6):
                msg = "Explicit interval boundaries do not continuously cover the requested horizon"
                raise ValueError(msg)
            result.append(weighted_total / duration)
        if present_value is not None:
            result[0] = present_value
        return result

    block = _build_strict_block(
        forecast_series,
        horizon_times[0],
        horizon_times[-1],
        interval_starts=True,
        allow_single_interval=allow_single_interval,
        allow_leading_present_interval=(
            present_value is not None and bool(np.isclose(forecast_series[0][0], horizon_times[1], rtol=0.0, atol=1e-6))
        ),
    )
    result: list[float] = []
    for interval_start, interval_end in pairwise(horizon_times):
        mask = (block["timestamp"] > interval_start) & (block["timestamp"] < interval_end)
        interval_points = block[mask]
        start_value = np.interp(interval_start, block["timestamp"], block["value"])
        end_value = np.interp(interval_end, block["timestamp"], block["value"])
        times = np.concatenate([[interval_start], interval_points["timestamp"], [interval_end]])
        values = np.concatenate([[start_value], interval_points["value"], [end_value]])
        result.append(float(np.trapezoid(values, times) / (interval_end - interval_start)))
    if present_value is not None:
        result[0] = present_value
    return result
