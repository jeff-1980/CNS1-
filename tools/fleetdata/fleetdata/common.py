"""Helpers shared by adapters."""
import re

import numpy as np
import pandas as pd


def snake(name):
    s = name.replace("°C", "degc").replace("°", "deg").replace("%", "pct").replace("/", "_per_")
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_").lower()
    return s or "col"


def unique_names(names):
    seen, out = {}, []
    for n in names:
        k = n
        i = 2
        while k in seen:
            k = f"{n}_{i}"
            i += 1
        seen[k] = 1
        out.append(k)
    return out


def map_core(raw_names, rules):
    """rules: list of (unified_name, regex). First raw name matching each regex wins; each raw name used once.
    Returns {raw_name: unified_name}."""
    used, out = set(), {}
    for uni, rx in rules:
        r = re.compile(rx, re.I)
        for n in raw_names:
            if n not in used and r.search(n):
                out[n] = uni
                used.add(n)
                break
    return out


def wind_exposure(power_col="power_kw", wind_col="wind_ms", interval_h=1 / 6):
    """Daily exposure for wind turbines: energy (kWh), hours producing, mean wind speed, hours above 20 m/s."""

    def f(unit, ts):
        if power_col not in ts:
            return None
        day = ts["time"].dt.floor("D")
        p = ts[power_col]
        out = pd.DataFrame({
            "energy_kwh": p.clip(lower=0).groupby(day).sum() * interval_h,
            "run_hours": (p > 0).groupby(day).sum() * interval_h,
        })
        if wind_col in ts:
            w = ts[wind_col]
            out["wind_mean"] = w.groupby(day).mean()
            out["hours_wind_gt20"] = (w > 20).groupby(day).sum() * interval_h
        return out

    return f


WIND_CORE_GREENBYTE = [
    ("power_kw", r"^Power \(kW\)$"),
    ("wind_ms", r"^Wind speed \(m/s\)$"),
    ("rotor_rpm", r"^Rotor speed \(RPM\)$"),
    ("gen_rpm", r"^Generator RPM \(RPM\)$"),
    ("pitch_deg", r"^Blade angle \(pitch position\)( A)? \(°\)$"),
    ("nacelle_pos_deg", r"^Nacelle position \(°\)$"),
    ("wind_dir_deg", r"^Wind direction \(°\)$"),
    ("ambient_c", r"^Nacelle ambient temperature \(°C\)$"),
    ("nacelle_c", r"^Nacelle temperature \(°C\)$"),
    ("gear_oil_c", r"^Gear oil temperature \(°C\)$"),
    ("gear_oil_inlet_c", r"^Gear oil inlet temperature \(°C\)$"),
    ("rotor_bearing_c", r"^Rotor bearing temp \(°C\)$"),
    ("front_bearing_c", r"^Front bearing temperature \(°C\)$"),
    ("rear_bearing_c", r"^Rear bearing temperature \(°C\)$"),
    ("gen_bearing_front_c", r"^Generator bearing front temperature \(°C\)$"),
    ("gen_bearing_rear_c", r"^Generator bearing rear temperature \(°C\)$"),
    ("stator_c", r"^Stator temperature 1 \(°C\)$"),
    ("power_setpoint_kw", r"^Turbine Power setpoint \(kW\)$"),
    ("drivetrain_acc", r"^Drive train acceleration \(mm/s2\)$"),
    ("metal_particles", r"^Metal particle count$"),
    ("energy_export_kwh", r"^Energy Export \(kWh\)$"),
    ("lost_production_kwh", r"^Lost Production to Downtime \(kWh\)$"),
]

WIND_CORE_HOT = [
    ("power_kw", r"^wtc_ActPower_mean$"),
    ("wind_ms", r"^wtc_AcWindSp_mean$"),
    ("rotor_rpm", r"^wtc_MainSRpm_mean$"),
    ("gen_rpm", r"^wtc_GenRpm_mean$"),
    ("pitch_deg", r"^wtc_PitcPosA_mean$|^wtc_PitchRef_BladeA_mean$"),
    ("yaw_deg", r"^wtc_YawPos_mean$"),
    ("nacelle_pos_deg", r"^wtc_NacelPos_mean$"),
    ("ambient_c", r"^wtc_AmbieTmp_mean$"),
    ("nacelle_c", r"^wtc_NacelTmp_mean$"),
    ("gear_oil_c", r"^wtc_GeOilTmp_mean$|^wtc_GearOilTmp_mean$"),
    ("gear_bearing_c", r"^wtc_GeBe\w*Tmp_mean$|^wtc_Gear\w*BeTmp_mean$"),
    ("main_bearing_c", r"^wtc_MainBTmp_mean$|^wtc_MaBe\w*Tmp_mean$"),
    ("gen_bearing_c", r"^wtc_GenBe\w*Tmp_mean$|^wtc_GeBeD\w*Tmp_mean$"),
    ("gen_winding_c", r"^wtc_GenWi\w*Tmp_mean$|^wtc_HSGenTmp_mean$"),
    ("power_setpoint_kw", r"^wtc_PowerRef_endvalue$"),
]

CARE_CORE = [  # matched against feature_description "description" of *_avg columns
    ("power_kw", r"^(ABB-LS )?Active power( HV grid)?$|^Grid power$|^Power$|active power.*grid side|^Total active power"),
    ("wind_ms", r"^Wind ?speed$|^Windspeed$|wind speed anemometer 1|^Wind speed"),
    ("rotor_rpm", r"rotor.*(speed|rpm)"),
    ("gen_rpm", r"generator.*(rpm|speed)"),
    ("ambient_c", r"^ambient temperature|^outside temperature"),
    ("pitch_deg", r"^pitch angle"),
]


def greenbyte_header(lines):
    """Find the CSV header in a Greenbyte export: the first line that is not a '#' comment, or the '# Date and time'
    line. Returns (index of header line, list of column names)."""
    import csv
    for i, ln in enumerate(lines):
        s = ln.strip("﻿\r\n")
        if s.startswith("# Date and time") or (s and not s.startswith("#")):
            s = s[2:] if s.startswith("# ") else s
            return i, next(csv.reader([s]))
    raise ValueError("no header found")


def is_stat_column(name):
    return bool(re.search(r", (standard deviation|minimum|maximum|stddev|stdev|std|min|max)\b", name, re.I))


def canon_raw(name):
    """Unify spelling differences between yearly exports of the same signal (unit strings)."""
    return name.replace("(mm/ss)", "(mm/s2)").replace("(mm/s²)", "(mm/s2)")


def coalesce_duplicate_columns(df):
    """After a many-to-one rename: merge same-named columns, first non-missing value wins (left to right)."""
    if not df.columns.duplicated().any():
        return df
    out = {}
    for name in dict.fromkeys(df.columns):
        block = df.loc[:, df.columns == name]
        out[name] = block.iloc[:, 0] if block.shape[1] == 1 else block.bfill(axis=1).iloc[:, 0]
    return pd.DataFrame(out, index=df.index)


def to_bool(x):
    return np.where(pd.isna(x), None, x.astype(str).str.strip().isin(["1", "True", "true", "yes", "Yes"]))
