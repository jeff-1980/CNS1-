## R1 rank stability (1000 joint draws)
T13 rank 1: E 78.7%, R_pp 92.5%, D3 91.2%; corrected E, R, D3 > 0 in 100% of draws.

## R2 pseudo-retrofit dates (pre-retrofit period; pre = 2020-01-01..date-8d, post = date..min(date+290d, 2021-09-22))
T13 corrected at pseudo dates: 2020-07-01 E -0.0008 R -0.408 D3 -0.0009 W -0.0002 (rankE 12, rankR 20); 2020-09-01 E 0.0021 R -0.428 D3 -0.0002 W 0.0031 (9,19); 2020-11-01 E -0.0051 R 0.023 D3 0.0003 W -0.0018 (13,10); 2021-01-01 E -0.0167 R -0.374 D3 -0.0062 W -0.0046 (17,18); 2021-03-01 E -0.0413 R -0.837 D3 -0.0236 W -0.0095 (20,21).
Null over 105 turbine-date pairs (each turbine corrected by median of the other 20): E sd 0.0280, 95th pct 0.0459, max 0.1145, share >= real T13 (0.0481) 0.0381; R_pp sd 0.317, 95th 0.448, max 0.654, share >= 1.666: 0; D3 sd 0.0137, 95th 0.0222, max 0.0385, share >= 0.0230: 0.0476.
+4% injected into T13 at pseudo dates, recovered minus true: E -0.0009, +0.0022, -0.0052, -0.0172, -0.0426; R_pp -0.69, -0.64, -0.04, -0.41, -0.84; D3 -0.0012, -0.0005, +0.0002, -0.0064, -0.0244; C_true 0.82, C_rec 0.80, 0.75, 0.99, 1.35, -0.38.

## R3 per-turbine K1 recovery error (real window, from run_out/D_known_truth.csv, 20 turbines)
E: median -0.0004, IQR -0.0195..0.0091, range -0.0313..0.0281, sd 0.0183; R_pp: sd 0.270, range -0.49..0.47; D3: sd 0.0071, range -0.0135..0.0118.
Placebo-calibrated intervals (T13 corrected minus each placebo's corrected effect, min..max over 20): E 0.022..0.077; R_pp 1.17..2.14; D3 0.0115..0.0356; W -0.0028..0.0327; C(m=3) 0.28..0.89 (18 of 20 within 0.44..0.59).

## R4 mechanism
Speed share W/E 0.345; mean torque T1/E 0.828. C(m=3): pure scaling 0.79; measured gain profile at fixed speed 0.67; + K w^2 speed response 0.58; measured 0.48.
Net relative uplift by reference-power bin (kW lower edge): 0 0.053, 200 0.100, 400 0.105, 600 0.100, 800 0.072, 1000 0.064, 1200 0.061, 1400 0.060, 1600 0.026, 1800 0.011, 2000 0.007, 2200 0.000.

## R5 control signatures (step = T13-minus-fleet, Oct21-May22 minus Jan-Aug21; rank of |step| among 21)
rpm p99 at rated -0.0021 (rank 13; fleet median |step| 0.0027); rpm median at rated -0.0002 (19); power at rated -1.65 kW (14; 3.41); set-point share <2290 +0.0005 (19); part-load pitch median 0.0000 (11); anemometer ratio to fleet +0.0993 (rank 1; fleet 0.0182).

## R5b alarm log (paper_aeroup_alarms.py db2d77a + read-only per-code check)
Total alarm records, T13 vs fleet median (rank of 21): before 60 d 10215 vs 9930.5 (9); works week 346 vs 405 (16); after 60 d 2525 vs 3072 (16); Jan-Aug 2021 25798 vs 25264.5 (9); Oct21-May22 12283 vs 13666.5 (13).
Codes elevated on T13 near works (59/60/61/65/67/68/69/92) occur on almost all turbines since 2020; T13 first occurrences 2020-02-12..2020-07-17; similar or higher 2021 counts on unmodified turbines (e.g. T21, T01, T03, T12).
8186/8187: T13 first 2021-09-24 (works week); T01 first 2022-07-23 (= works start), T15 2022-07-15 (works start 07-14) -> works-stop codes.
8110: T13 first 2021-10-02, 0/46/5/2 per year 2020-23; also on 9 other turbines in bursts (T06 2021, T11 2023 ...).
Alarm description file covers only 12 codes; none of the codes above described.
