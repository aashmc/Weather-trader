# Late Session Guard (Plain English)

## What changed

The bot now has a city-by-city late-entry guard to reduce low-quality trades late in the market day.

It does three things:

1. Learns a local cutoff hour per city from last 365 days of hourly temperatures.
2. After cutoff, only allows the market-favorite bracket and uses stricter thresholds.
3. Near day end ("freeze window"), blocks all new entries.

It also blocks brackets that are already impossible once observed day-high has passed them.

## Why this helps

Late in the day, the daily max is often already set.  
Without this guard, model edge can look positive on brackets that are no longer realistic.

## New files / wiring

- `late_guard.py`: historical peak-hour fetch + cutoff cache + runtime context
- `bot.py`: builds `late_context` for each city/date and passes it to strategy
- `strategy.py`: applies late guard filters + impossible-bracket block
- `logger.py`: logs late-guard phase/cutoff/source fields
- `index.html`: dashboard now shows normal/late/freeze status and applies same late filter logic

## Main config knobs (`config.py`)

- `LATE_GUARD_ENABLED`
- `LATE_GUARD_CUTOFF_QUANTILE` (default `0.80`)
- `LATE_GUARD_FALLBACK_CUTOFF_HOURS`
- `LATE_GUARD_AFTER_CUTOFF_FAVORITE_ONLY`
- `LATE_GUARD_AFTER_CUTOFF_MIN_EDGE` (default `0.08`)
- `LATE_GUARD_AFTER_CUTOFF_FAMILY_BUMP` (default `+1`)
- `LATE_GUARD_FREEZE_HOURS_AFTER_CUTOFF` (default `6`)
- `LATE_GUARD_FREEZE_ALL_NEW_ENTRIES`

## Logged fields added

Cycle rows now include:

- `v5_late_phase`
- `v5_late_applies`
- `v5_late_cutoff`
- `v5_late_freeze`
- `v5_late_source`
- `v5_late_samples`
- `v5_obs_day_high`

Snapshots also store full `v5_filters.late_guard`.
