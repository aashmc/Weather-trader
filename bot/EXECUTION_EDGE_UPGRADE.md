# Execution-Aware Edge Upgrade (Plain English)

This note explains what changed in the bot.

## Why this was added

Before this update, `true_edge` was:

`corrected_prob - ask`

That ignores:
- whether the order fully fills in 5 minutes
- partial fills
- execution friction (fee/gas/slippage assumptions)

So paper edge could look better than real edge.

## What `true_edge` means now

The bot now estimates:

1. `effective_ask`
- starts from best ask
- adds optional fee/gas
- adds a spread penalty when most of the order is unlikely to fill immediately

2. `fill_fraction`
- expected fraction of contracts that will fill inside timeout
- based on:
  - top ask depth (immediate fill)
  - total ask depth
  - spread (wider spread = worse fill odds)

3. `edge_after_costs`
- `corrected_prob - effective_ask`

4. `true_edge` (used by filters)
- `edge_after_costs * fill_fraction`

This means low fill chance directly reduces edge.

## Kelly sizing change

Kelly sizing now uses:
- `effective_ask` (instead of raw ask) when execution-aware mode is on
- extra multiplier by `fill_fraction`

So size is naturally reduced when execution quality is poor.

## New config flags

In `config.py`:

- `EXECUTION_ADJUSTED_EDGE_ENABLED = True`
- `EXECUTION_FEE_RATE = 0.0`
- `EXECUTION_GAS_USD_PER_ORDER = 0.0`
- `EXECUTION_SPREAD_REF = 0.04`
- `EXECUTION_QUEUE_PENALTY = 0.50`
- `EXECUTION_MIN_FILL = 0.05`
- `EXECUTION_MAX_FILL = 0.98`

If you want to go back to old behavior, set:

`EXECUTION_ADJUSTED_EDGE_ENABLED = False`

## New fields saved in logs

Per bracket, snapshots now include:
- `effective_ask`
- `fill_fraction`
- `immediate_fill_fraction`
- `continuation_fill_prob`
- `fee_per_contract`
- `gas_per_contract`
- `slippage_penalty`
- `edge_after_costs`

## Important note

This is still a model estimate (not a perfect market simulator).
It is a safer approximation than raw ask-based edge, but it is not a full matching-engine replay.

## Dashboard alignment

`index.html` now uses the same execution-aware idea for displayed `true edge`:

- computes `effective ask`
- estimates `fill fraction`
- shows `true edge = edge_after_costs * fill_fraction`

In the signal card you now see:
- Ask
- Eff Ask
- Fill
- True Edge

Current dashboard execution defaults are kept close to bot behavior:
- fee rate = 0
- gas per order = 0 (unless you enable `useLiveGasEstimate`)

So dashboard and bot logic are now aligned conceptually and numerically.
