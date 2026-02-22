# Weather Trading System — Model Reference & Bias Calibration

**Last updated:** 2026-02-22
**Calibration window:** 2026-01-08 to 2026-02-21 (45 days)
**Ground truth source:** METAR station observations via IEM (Iowa Environmental Mesonet)
**Why METAR:** Polymarket resolves on Wunderground readings, which report METAR station data. This is the only valid ground truth for our edge calculations.

---

## Important Finding: Circular Comparison Trap

Our initial calibration compared models against Open-Meteo's archive observations. This was **invalid** because Open-Meteo's archive is partially derived from ERA5/ECMWF reanalysis — comparing ECMWF forecasts against ECMWF-derived "observations" made ECMWF look artificially accurate.

Switching to independent METAR station data changed rankings significantly, especially for NYC where ECMWF went from appearing best (0.57 RMSE) to 5th place (3.8°F RMSE).

**Lesson:** Always validate against the actual resolution source.

---

## Available Ensemble Models on Open-Meteo

| Model | Members | Resolution | Coverage | Notes |
|-------|---------|------------|----------|-------|
| ECMWF IFS Ensemble | 50 | 25km, 3-hourly | Global | Best overall, but has location-specific biases |
| ECMWF AIFS Ensemble | 50 | 25km, 6-hourly | Global | AI model, no historical accuracy data yet |
| ICON EPS | 39 | 26km, hourly | Global | Best for NYC in our tests |
| GEM Global Ensemble | 20 | ~25km | Global | Consistent mid-tier performance |
| UKMO Global Ensemble | 17 | 20km | Global | Worst for Seoul/Seattle, OK for London |
| BOM ACCESS Ensemble | 17 | ~25km | Global | No historical accuracy data available |

### Models WITHOUT ensemble (single forecast only)
| Model | Coverage | Notes |
|-------|----------|-------|
| KMA (Korea) | Global + 1.5km Korea | Best for Seoul (0.9°C RMSE) but no ensemble members |
| JMA (Japan) | Global | Poor for NYC/Seoul |
| GFS (NOAA) | Global | Ensemble endpoint broken on Open-Meteo as of Feb 2026 |

### GFS Ensemble Bug
As of Feb 2026, `gfs_seamless_eps`, `gfs025_eps`, `gfs05_eps` all return HTTP 400 for any location on the Open-Meteo Ensemble API. The deterministic GFS (`ncep_gfs013`) works on the Forecast API. Our old code used `gfs_seamless` which silently mapped to `ncep_gefs_seamless` (30 members), but this model name is unreliable.

### Old Code Bug
Previous code used model string `gfs_seamless,ecmwf_ifs,icon_seamless,gem_global` on the ensemble endpoint. This gave only 89 members (ICON 39 + GFS 30 + GEM 20). **ECMWF was completely missing** (wrong model name — should be `ecmwf_ifs025_ensemble`). Since ECMWF is the world's best weather model, this was a major gap.

---

## Per-City Model Selection

### London (EGLC) — 176 ensemble members

**Models used:** ECMWF IFS (50) + ECMWF AIFS (50) + ICON (39) + GEM (20) + UKMO (17)
**Dropped:** None — all models perform well for London

| Model | RMSE | MAE | Bias | SD | Verdict |
|-------|------|-----|------|----|---------|
| KMA | 0.5°C | 0.4°C | +0.2°C | 0.5°C | No ensemble |
| ICON | 0.6°C | 0.4°C | −0.3°C | 0.5°C | ✅ Included |
| JMA | 0.7°C | 0.5°C | +0.2°C | 0.6°C | No ensemble |
| ECMWF IFS | 0.7°C | 0.5°C | +0.3°C | 0.6°C | ✅ Included |
| UKMO | 0.7°C | 0.6°C | +0.4°C | 0.6°C | ✅ Included |
| GFS | 0.8°C | 0.6°C | +0.3°C | 0.7°C | Ensemble broken |
| GEM | 0.8°C | 0.6°C | +0.2°C | 0.8°C | ✅ Included |

**Combined bias correction:** mean=+0.17°C, sd=0.66°C
**Interpretation:** Models underpredict by 0.17°C on average. Tiny correction.

---

### Seoul (RKSI) — ~130 members (120 ensemble + 10 KMA synthetic)

**Models used:** ECMWF IFS (50) + ECMWF AIFS (50) + GEM (20) + KMA synthetic (10)
**Dropped:** ICON (1.7 RMSE), UKMO (2.8 RMSE), GFS (2.4), JMA (2.2)

| Model | RMSE | MAE | Bias | SD | Verdict |
|-------|------|-----|------|----|---------|
| KMA | 0.9°C | 0.7°C | +0.4°C | 0.8°C | ✅ Synthetic (10 members) |
| GEM | 1.0°C | 0.8°C | +0.3°C | 1.0°C | ✅ Included |
| ECMWF IFS | 1.1°C | 0.9°C | +0.8°C | 0.8°C | ✅ Included |
| ICON | 1.7°C | 1.3°C | +0.7°C | 1.6°C | ❌ Mediocre |
| JMA | 2.2°C | 1.6°C | +0.3°C | 2.2°C | ❌ Poor |
| GFS | 2.4°C | 1.8°C | +0.1°C | 2.4°C | ❌ Poor |
| UKMO | 2.8°C | 2.0°C | +1.2°C | 2.6°C | ❌ Awful |

**Combined bias correction:** mean=+0.69°C, sd=0.86°C
**Interpretation:** Models underpredict by 0.69°C. The bad models (ICON/UKMO/GFS/JMA) had huge scatter — removing them dramatically reduces our SD from 2.0 to 0.86.

**KMA synthetic approach:** KMA gives a single deterministic forecast. We add 10 members with ±0.3° noise around it. This gives Korea's best local model a voice without dominating the 120 real ensemble members.

---

### NYC (KLGA) — 89 ensemble members

**Models used:** ICON (39) + ECMWF IFS (50)
**Dropped:** GEM (4.0 RMSE), JMA (4.6), UKMO (2.7)
**ECMWF AIFS:** Excluded — likely shares ECMWF IFS's large +2.8°F bias, unverifiable

| Model | RMSE | MAE | Bias | SD | Verdict |
|-------|------|-----|------|----|---------|
| ICON | 1.7°F | 1.3°F | +0.5°F | 1.6°F | ✅ Best by far |
| KMA | 2.0°F | 1.5°F | +0.3°F | 2.0°F | No ensemble |
| UKMO | 2.7°F | 2.2°F | +1.6°F | 2.2°F | ❌ Mediocre |
| GFS | 3.2°F | 2.8°F | +1.9°F | 2.6°F | ❌ Poor |
| ECMWF IFS | 3.8°F | 3.2°F | +2.8°F | 2.6°F | ✅ Included (bias-corrected) |
| GEM | 4.0°F | 3.4°F | +3.1°F | 2.6°F | ❌ Poor |
| JMA | 4.6°F | 4.0°F | +3.8°F | 2.6°F | ❌ Awful |

**Combined bias correction:** mean=+1.79°F, sd=2.49°F
**Interpretation:** Models underpredict by almost 2°F. ECMWF alone is off by 2.8°F but included because it adds 50 members and the bias is consistent (correctable). NYC has the highest uncertainty (SD=2.49°F) — signals here are less confident.

**Why ECMWF is kept despite high bias:** The bias is systematic and consistent (+2.8°F over 45 days). After correction, the 50 members add valuable spread information. A consistent bias is correctable; random noise is not.

---

### Seattle (KSEA) — 109 ensemble members

**Models used:** ICON (39) + ECMWF IFS (50) + GEM (20)
**Dropped:** UKMO (2.9 RMSE), JMA (2.6)
**ECMWF AIFS:** Excluded — same bias concern as ECMWF IFS, unverifiable

| Model | RMSE | MAE | Bias | SD | Verdict |
|-------|------|-----|------|----|---------|
| KMA | 1.9°F | 1.5°F | +1.2°F | 1.4°F | No ensemble |
| GFS | 1.9°F | 1.5°F | +0.4°F | 1.9°F | Ensemble broken |
| ICON | 2.4°F | 1.8°F | +1.6°F | 1.8°F | ✅ Included |
| ECMWF IFS | 2.6°F | 2.0°F | +1.7°F | 2.0°F | ✅ Included |
| GEM | 2.6°F | 2.1°F | +1.7°F | 2.0°F | ✅ Included |
| JMA | 2.6°F | 2.2°F | +1.5°F | 2.2°F | ❌ Marginal |
| UKMO | 2.9°F | 2.4°F | +2.3°F | 1.7°F | ❌ Worst |

**Combined bias correction:** mean=+1.66°F, sd=1.93°F
**Interpretation:** All three included models have nearly identical bias (+1.6–1.7°F). When models agree this much, the correction is reliable. 109 members gives good spread.

---

## Bias Correction Summary

| City | Station | Members | Bias | SD | Unit | Method |
|------|---------|---------|------|-----|------|--------|
| London | EGLC | 176 | +0.17 | 0.66 | °C | Monte Carlo N=50k |
| Seoul | RKSI | ~130 | +0.69 | 0.86 | °C | Monte Carlo N=50k |
| NYC | KLGA | 89 | +1.79 | 2.49 | °F | Monte Carlo N=50k |
| Seattle | KSEA | 109 | +1.66 | 1.93 | °F | Monte Carlo N=50k |

**Bias direction:** Positive = station reads HIGHER than model predicts = models underpredict
**Correction effect:** Shifts probability distribution toward higher brackets
**All biases positive:** METAR stations consistently record higher daily maxima than models predict, likely because stations capture brief peaks that gridded models smooth out.

---

## Calibration Methodology

1. **Source hourly METAR** from IEM (mesonet.agron.iastate.edu) for each station
2. **Compute daily max** in local timezone from hourly observations
3. **Convert units** for °F markets (METAR always reports °C)
4. **Fetch model forecasts** from Open-Meteo Historical Forecast API (`past_days=45`)
5. **Compute error** = METAR daily max − model daily max
6. **Aggregate** mean bias, SD, RMSE, MAE per model per city
7. **Weight** by ensemble member count for combined bias

**Calibration script:** `/home/claude/calibrate_bias.py`
**METAR data cache:** `/home/claude/metar_daily_max.json`

### Known Bug Fixed (Feb 22, 2026): METAR Timezone

The original daily max computation matched METAR observations using the UTC day from raw observation strings (e.g., `RKSI 221500Z` → day 22 in UTC). This caused timezone-dependent errors:

- **Seoul (UTC+9):** Midnight-9am readings had previous UTC day → missed from daily max. Feb 22 showed 8°C instead of actual 12°C (4°C error).
- **NYC (UTC-5):** Evening readings after 7pm had next UTC day → missed from daily max.
- **London (UTC+0):** No impact (UTC and local day align).

**Fix:** Replaced UTC-day regex with `toLocaleDateString('en-CA', {timeZone: CITY.tz})` using `obsTime` epoch seconds. This correctly computes the calendar-day maximum in the station's local timezone.

**Additional fix — Midnight boundary:** Wunderground treats the 00:00 observation as the *end* of the previous day, not the start of the new day. Example: Seoul Feb 22 METAR shows 12°C at 00:00 KST, but Wunderground assigns it to Feb 21. The actual Feb 22 max excluding midnight = 11°C, matching Polymarket's resolution. The filter now excludes exact `00:00:00` local time readings.

---

## Re-Calibration Schedule

Bias values should be re-calibrated:
- **Monthly** — as seasons change, model biases shift
- **After major model updates** — ECMWF/GFS upgrades happen periodically
- **If PnL diverges from expected** — systematic losses suggest bias drift

To re-calibrate: re-run the 45-day comparison against fresh METAR data and update the `bias` config in `CITIES`.

---

## Competitor Analysis: NOAA/OpenClaw/Simmer Bots

As of Feb 2026, there are automated bots trading Polymarket weather using NOAA/GFS single-point forecasts (OpenClaw + Simmer SDK). Key differences from our system:

| Feature | Our System | NOAA Bots |
|---------|-----------|-----------|
| Models | Multi-model ensemble (89–176 members) | Single NOAA/GFS forecast |
| Bias correction | METAR-calibrated per city | None |
| Sizing | Kelly criterion based on edge | Fixed thresholds (buy <15¢, sell >45¢) |
| Order book | Volume-weighted average price | No slippage modeling |
| Probability | Full distribution from ensemble | Single point estimate |
| Model accuracy | ECMWF+ICON (best available) | GFS only (3rd–6th best depending on city) |

**Our edge:** Probability distribution + bias correction + city-specific model selection vs their single uncorrected forecast.
