# Strava Bike Selector

Listens for new Strava rides via webhook, predicts which of your bikes you rode
with an XGBoost classifier trained on your own history, sets the gear on the
activity, and appends the probabilities to the description:

```
Sunday loop with the club.

Bike guess: Canyon Endurace 87% | Cube Nuroad 11% | Rose Commuter 2% [bike-selector]
```

Built to run on a free Render web service. Python 3.11, `uv`, FastAPI,
[stravalib](https://stravalib.readthedocs.io/), XGBoost.

---

## How it works

```
Strava upload
     │
     ▼  POST /webhook/<secret>   (must answer 200 within 2 s)
FastAPI  ──ack immediately──▶ BackgroundTask
                                   │
                                   ├─ fetch the activity (stravalib)
                                   ├─ ensure model  ──▶ cached in /tmp, else
                                   │                    train on ride history
                                   ├─ extract ~44 features → predict_proba
                                   └─ PUT gear_id + description
```

**No database.** Your refresh token lives in an environment variable and the
model is trained from the Strava API on boot, then cached in `/tmp`. Nothing to
provision, nothing to pay for.

### Features the model uses

Everything comes from the *summary* activity payload, so training is one
paginated API call rather than one request per ride.

| Group | Features |
| --- | --- |
| Effort | `avg_watts`, `max_watts`, `weighted_avg_watts`, `kilojoules`, `watts_per_kmh` |
| Speed | `avg_speed_kmh`, `max_speed_kmh`, `speed_burstiness` |
| Distance & time | `distance_km`, `moving_min`, `elapsed_min`, `moving_ratio` |
| Terrain | `elev_gain_m`, `elev_per_km`, `elev_high_m`, `elev_low_m`, `elev_range_m` |
| Sensors | `has_power_meter`, `has_heartrate`, `has_cadence`, `avg_cadence`, `avg_heartrate`, `max_heartrate`, `avg_temp_c` |
| Context | `is_trainer`, `is_commute`, `is_manual`, `start_lat`, `start_lng`, `start_hour`, `day_of_week`, `is_weekend`, `month` |
| Sport | `sport_Ride` (always 1 — only sport_type `Ride` is eligible, see below) |

Missing values are passed to XGBoost as `NaN`, not zero. That is deliberate:
"this ride had no power meter" is one of the strongest signals about which bike
you were on, and XGBoost learns a dedicated branch for missing values.

Two other details worth knowing:

- **Only `sport_type == "Ride"` is eligible.** Rides tagged `VirtualRide`,
  `GravelRide`, `MountainBikeRide`, `EBikeRide`, etc. are skipped entirely —
  not trained on, not predicted for, not auto-tagged. Virtual rides are always
  done on the same (trainer) bike, so they carry no signal; the other `Ride`
  subtypes are excluded too so the eligible set is exactly `Ride`.
- **Recency weighting.** Rides are weighted by `0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)`
  and by inverse class frequency, so a bike you bought last spring is not
  drowned out by five years of riding something else.
- **No feedback loop.** Activity ids this app labelled itself are recorded and
  excluded from later training runs, so the model never learns from its own
  guesses. That list lives in `/tmp` and is lost on redeploy — see
  [Caveats](#caveats).

---

## Setup

### 1. Create a Strava API application

At <https://www.strava.com/settings/api>, note the **Client ID** and **Client
Secret**, and set **Authorization Callback Domain** to `localhost` for now.

### 2. Install and configure

```bash
git clone <your-repo> bike-selector && cd bike-selector
uv sync
cp .env.example .env
```

Fill in `STRAVA_CLIENT_ID` and `STRAVA_CLIENT_SECRET`, then invent random
strings for `WEBHOOK_VERIFY_TOKEN` and `WEBHOOK_PATH_SECRET`.

### 3. Authorise

```bash
uv run bike-selector auth
```

This opens Strava in your browser, catches the redirect on
`http://localhost:8721/exchange`, and prints a refresh token. Paste it into
`.env` as `STRAVA_REFRESH_TOKEN`.

The requested scopes are `read`, `activity:read_all`, `activity:write` and
`profile:read_all` — `activity:write` is what lets the app change the gear.

### 4. Check the model before deploying anything

```bash
uv run bike-selector bikes      # confirm the app can see your bikes
uv run bike-selector train      # cross-validated accuracy report
uv run bike-selector predict 12345678901   # probabilities for one ride, no writes
```

`train` prints something like:

```
Trained on 842 rides across 4 bikes (2026-08-02 09:14 UTC)
5-fold CV accuracy: 91.3% (always-guess-most-common baseline: 54.1%)
  Canyon Endurace                    456 rides
  Cube Nuroad                        201 rides
  Rose Commuter                      142 rides
  Santa Cruz Hightower                43 rides
Most informative features:
  has_power_meter                     0.211
  distance_km                         0.164
  avg_speed_kmh                       0.131
  ...
```

If CV accuracy is not comfortably above the baseline, the model is not learning
anything useful yet — usually because too few past rides have a bike attached.
Fix that in Strava first.

### 5. Deploy to Render

Push to GitHub, then **New → Blueprint** and point Render at `render.yaml`.
Set the secret env vars in the dashboard (`STRAVA_CLIENT_ID`,
`STRAVA_CLIENT_SECRET`, `STRAVA_REFRESH_TOKEN`, `WEBHOOK_VERIFY_TOKEN`,
`WEBHOOK_PATH_SECRET`) and update `PUBLIC_BASE_URL` to your real service URL.

Verify: `curl https://<your-app>.onrender.com/healthz`.

### 6. Register the webhook

Only after the service is live and reachable:

```bash
PUBLIC_BASE_URL=https://<your-app>.onrender.com uv run bike-selector webhook subscribe
uv run bike-selector webhook list
```

Strava immediately GETs your callback and expects the `hub.challenge` echoed
within two seconds. Strava allows exactly one subscription per API application —
`webhook delete <id>` first if one already exists.

Now upload a ride.

---

## The free-tier cold-start problem

Render's free instances spin down after ~15 minutes without inbound traffic, and
the cold start takes far longer than Strava's two-second webhook budget. Strava
retries three times and then drops the event.

Two mitigations ship with this app:

1. **Self-ping.** `KEEPALIVE_MINUTES=10` makes the app request its own
   `/healthz` every ten minutes, which counts as inbound traffic. This burns
   ~4,300 of the 750 free instance-hours per month... which is to say it keeps
   one service running continuously, and Render's free tier allows exactly one.
   For something more robust, point [cron-job.org](https://cron-job.org) at
   `/healthz` instead — an external pinger survives the app itself crashing.
2. **Backfill.** If an event is lost anyway, nothing is permanently missed:

   ```bash
   uv run bike-selector backfill --days 30            # dry run
   uv run bike-selector backfill --days 30 --no-dry-run
   ```

   This scans recent rides that still have no bike attached and processes them.

---

## Configuration

Everything is environment variables; see `.env.example` for the full list.

| Variable | Default | Effect |
| --- | --- | --- |
| `SET_GEAR` | `true` | Assign the winning bike |
| `WRITE_DESCRIPTION` | `true` | Append the probability line |
| `OVERWRITE_EXISTING_GEAR` | `true` | Reassign even when a bike is already set |
| `MIN_CONFIDENCE` | `0.0` | Below this, annotate but do not assign |
| `SKIP_TRAINER` | `false` | Ignore indoor rides |
| `MIN_RIDES_PER_BIKE` | `5` | Bikes with less history are excluded as classes |
| `RECENCY_HALF_LIFE_DAYS` | `365` | How fast old rides lose influence |
| `MODEL_TTL_HOURS` | `24` | Retrain when the cache is older than this |
| `MAX_PROBABILITIES_SHOWN` | `3` | Bikes listed in the description |

Want it to suggest rather than decide? Set `MIN_CONFIDENCE=0.75` and
`OVERWRITE_EXISTING_GEAR=false`.

---

## Endpoints

| Route | Purpose |
| --- | --- |
| `GET /healthz` | Liveness + whether a model is loaded |
| `GET /model` | Training report, CV accuracy, feature importances |
| `POST /model/retrain` | Force a retrain in the background |
| `GET /recent` | Last 25 processed events (in-memory ring buffer) |
| `POST /activities/{id}/predict` | Manual run; `?dry_run=false` to write |
| `GET/POST /webhook/<secret>` | Strava subscription validation + events |

---

## Development

```bash
uv sync                     # includes the dev group
uv run pytest               # ~50 tests, no network or credentials needed
uv run ruff check . && uv run ruff format .
uv run bike-selector serve --reload
```

The test suite runs against synthetic ride histories in `tests/conftest.py` —
three bikes with distinct but deliberately overlapping and noisy signatures — so
`test_cross_validated_accuracy_beats_baseline` genuinely fails if the feature
pipeline regresses.

VS Code launch configurations for the server and the CLI are in `.vscode/`.
Install the Ruff extension; format-on-save and import sorting are pre-wired.

To exercise the webhook locally, tunnel it:

```bash
uv run bike-selector serve            # terminal 1
ngrok http 8000                       # terminal 2
PUBLIC_BASE_URL=https://<id>.ngrok-free.app uv run bike-selector webhook subscribe
```

---

## Caveats

- **`/tmp` is ephemeral.** Redeploying loses the cached model (it retrains on
  boot, ~30 s) and the list of self-labelled activity ids. After a redeploy the
  model can start learning from its own past guesses. If that bothers you, move
  `STATE_DIR` onto a Render persistent disk, or swap the two small JSON files
  for a free Neon/Supabase Postgres.
- **Rotated refresh tokens.** Strava usually returns the same refresh token, but
  when it rotates one the new value is cached in `STATE_DIR` and logged as a
  warning. Copy it into `STRAVA_REFRESH_TOKEN` when you see that warning, or the
  app breaks on the next redeploy.
- **Memory.** XGBoost + numpy + scikit-learn on Render's 512 MB free plan is
  workable but not roomy; that is why pandas is not a dependency and why
  `OMP_NUM_THREADS=1` is set. scikit-learn (and its scipy dependency) is
  required by `XGBClassifier`, xgboost's sklearn-API wrapper — dropping it means
  porting `model.py` to the native `xgboost.train`/`Booster` API. If you hit
  OOM, lower `TRAINING_ACTIVITY_LIMIT`.
- **Probabilities are not calibrated.** With a few hundred rides XGBoost tends
  to be overconfident. Treat "87%" as a ranking score, not a true frequency.
  The cross-validated accuracy from `GET /model` is the honest number.
- **Strava does not send webhooks for gear changes** (only title, type and
  privacy), so the app's own writes cannot re-trigger it. No loop risk.
