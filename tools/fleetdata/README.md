# tools/fleetdata — one layout for "units × time series × event table" public datasets

Added 2026-11-05. Idea borrowed from ClawsGO Science's "pre-installed reference datasets": convert every fleet-type
dataset already downloaded on jeff's PC into the same on-disk layout once, so that a new research question starts
from `open_fleet(...)` instead of a new parser. CPU only, raw files are only read.

## Layout (`<out>/<dataset>/`)

| file | content |
|---|---|
| `meta.json` | licence, DOI/source, build time, time convention, interval, row counts, notes (assumptions, unmatched channels) |
| `units.parquet` | `unit_id, dataset, site, manufacturer, model, rated_kw, commission_date, lat, lon, meta` |
| `channels.parquet` | `channel, raw_name, table, unit_of_measure, core` (core = unified name shared across datasets) |
| `ts/<unit_id>.parquet` | `time` (UTC, end of the averaging interval) + float32 channels |
| `events.parquet` | `unit_id, start, end, kind, code, text, category, stopping, source` + dataset-specific `x_*` |
| `daily.parquet` | unit × day: `n_samples`, `coverage`, exposure (`energy_kwh`, `run_hours`, `wind_mean`, ...), `n_<kind>` |

Event `kind` vocabulary: `fault, alarm, stop_other, maintenance, intervention, inspection, anomaly, other`. The raw
status / category text is always kept, so a different classification can be applied later.

Core channel names used so far: `power_kw, wind_ms, rotor_rpm, gen_rpm, pitch_deg, yaw_deg, nacelle_pos_deg,
wind_dir_deg, ambient_c, nacelle_c, gear_oil_c, gear_oil_inlet_c, rotor_bearing_c, front_bearing_c,
rear_bearing_c, gen_bearing_front_c, gen_bearing_rear_c, gen_bearing_c, gen_winding_c, main_bearing_c, stator_c,
power_setpoint_kw, drivetrain_acc, metal_particles, energy_export_kwh, lost_production_kwh, shutdown_s,
status_type_id, cabinet_temp_c, setpoint_c, mains_v, state`. Each build's `meta.json` lists the core names it did not
find.

## Datasets (all built 2026-11-05 on jeff's PC, `--check` VALIDATE OK)

Built sizes / counts: kelmarsh 6 units, 133 channels (22 core), 503,470 events, 734 MB; penmanshiel 14 units, 127 (22), 1,056,488 events, 1.4 GB; hill_of_towie 21 units, 79 (14), 2,227,487 events, 1.3 GB; care_a/b/c 5/9/22 assets, 22/15/58 event windows; ult_revco 25 freezers, 12 (4), 97,432 events, 1.8 GB.

| name | source on jeff's PC | units | events | notes |
|---|---|---|---|---|
| `kelmarsh` | `~/wind/zips/kelmarsh` (Zenodo 8252025, CC BY 4.0) | 6 Senvion MM92 | Status logs | Greenbyte time = interval start (assumed), shifted +10 min; 2023–24 exports repeat timestamps ~40×, collapsed (first non-missing value per channel) — see `diag_greenbyte_dups.py` |
| `penmanshiel` | `~/wind/zips/penmanshiel` (Zenodo 8253010, CC BY 4.0) | 14 Senvion MM82 | Status logs | as Kelmarsh; WT01–10 lack Jan 2023 and 2024 |
| `hill_of_towie` | `D:\audit_data\hill_of_towie` (v2.1.0, CC BY 4.0), 2020–2023 zips | 21 Siemens SWT-2.3 | tblAlarmLog, AeroUp install dates | alarm description table covers only a few codes → most alarms `other` |
| `care_a/b/c` | `~/data_care` (CARE to Compare) | 5 / 9 / 22 assets | one anomaly/normal window per event dataset | time stamps anonymised by the publisher |
| `ult_revco` | `D:\audit_data\ult_freezer` (Sci Data 2023, CC0) | 24 Revco freezers | faults, service visits, controller events | irregular sampling, no time zone given |

Not included yet: UK MOT (no per-unit time series; the scan thread's cleaned parquet in `D:\audit_data\mot\work*`
is the reference), Backblaze (own repo), and the bench vibration sets (CWRU, PU, XJTU-SY, ...), which need a
different "segment × label × condition" layout.

## Use

```bash
cd ~/fleetdata                                   # copy of this folder on jeff's PC (WSL)
~/tfvenv/bin/python build.py --selftest          # synthetic round trip, must print SELFTEST PASS
~/tfvenv/bin/python build.py kelmarsh            # -> /mnt/d/audit_data/_unified/kelmarsh (+ kelmarsh.log)
~/tfvenv/bin/python build.py --check /mnt/d/audit_data/_unified/kelmarsh
~/tfvenv/bin/python build.py kelmarsh --headers  # list raw columns only
```

```python
import sys; sys.path.insert(0, "/home/jeffwork/fleetdata")
from fleetdata import open_fleet
f = open_fleet("/mnt/d/audit_data/_unified/kelmarsh")
f.units; f.events(kind="fault"); f.ts("KEL01", core=True); f.daily()
f.panel(["power_kw", "gear_oil_c"], freq="D")          # long unit × day panel
cfg = f.to_h2kit("/tmp/h2_kel", event_kinds=["fault"])   # files + YAML for h2kit (baseline-first H2 check)
```

`to_h2kit` writes per-unit series, `events.csv`, `avoid.json` (starts of `maintenance` events) and
`h2kit_config.yaml` for `tools/topic_forecast/h2kit/h2kit.py` (branch `claude/project-thread-kfswhq`).

`build.py --channels core|means|all`: `means` (default) keeps the core channels plus every 10-min mean column
present in all exports of a site; `core` only the unified channels; `all` everything.
