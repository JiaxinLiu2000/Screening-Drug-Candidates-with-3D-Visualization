# Binding-affinity baseline (Davis, KIBA)

RandomForestRegressor, n_estimators=100, max_features='sqrt', random_state=seed, n_jobs=-1; fit on train only (valid unused). Features: Morgan r=2 2048 bits + protein AAC(20)+DPC(400) = 2468 float32, full data (no subsampling). CI is exact over all comparable test pairs. Davis target = pKd (TDC convert_to_log of Kd nM); KIBA target = KIBA score.

## Dataset statistics

| dataset | raw pairs | pairs dropped (std.) | pairs merged (InChIKey) | drugs | proteins | pairs | density |
|---|---|---|---|---|---|---|---|
| DAVIS | 25772 | 0 | 0 | 68 | 379 | 25772 | 1.0000 |
| KIBA | 117657 | 413 | 51 | 2030 | 229 | 117193 | 0.2521 |

| dataset | units | min | p25 | median | p75 | max | mean | std |
|---|---|---|---|---|---|---|---|---|
| DAVIS | pKd | 5.0000 | 5.0000 | 5.0000 | 5.4231 | 9.9355 | 5.4074 | 0.8263 |
| KIBA | KIBA score | 0.0000 | 11.2000 | 11.5559 | 11.9500 | 17.2002 | 11.7227 | 0.8337 |

Davis: TDC returned Kd in nM (raw, no log transform); converted via pKd = -log10(Kd_nM*1e-9 + 1e-10) (TDC convert_to_log). Censored ceiling Kd = 10000 nM (pKd = 4.999996): 18343 raw rows, 18343 of 25772 final pairs (71.2%).

## Split sizes (identical across seeds unless shown as min-max)

| dataset | split | n_train | n_valid | n_test | test drugs | test targets |
|---|---|---|---|---|---|---|
| DAVIS | random | 20618 | 2577 | 2577 | 68 | 378-379 |
| DAVIS | cold_drug | 20466 | 2653 | 2653 | 7 | 379 |
| DAVIS | cold_target | 20604 | 2584 | 2584 | 68 | 38 |
| KIBA | random | 93754 | 11719 | 11720 | 1802-1825 | 224-228 |
| KIBA | cold_drug | 93091-94988 | 10587-12366 | 11618-12481 | 203 | 222-229 |
| KIBA | cold_target | 90565-98326 | 8794-12407 | 10073-15177 | 1714-1976 | 23 |

Zero train->valid/test entity leakage verified for every cold split (script asserts).

## Test metrics (mean ± sample std over seeds 0-4)

| dataset | split | RMSE | Pearson r | Spearman rho | CI |
|---|---|---|---|---|---|
| DAVIS | random | 0.6179 ± 0.0204 | 0.6773 ± 0.0093 | 0.6122 ± 0.0038 | 0.8429 ± 0.0044 |
| DAVIS | cold_drug | 0.7441 ± 0.0659 | 0.4530 ± 0.1897 | 0.3815 ± 0.2324 | 0.7038 ± 0.1220 |
| DAVIS | cold_target | 0.6489 ± 0.0469 | 0.5809 ± 0.0317 | 0.5369 ± 0.0208 | 0.8046 ± 0.0099 |
| KIBA | random | 0.5216 ± 0.0044 | 0.7841 ± 0.0044 | 0.7861 ± 0.0022 | 0.8241 ± 0.0012 |
| KIBA | cold_drug | 0.5708 ± 0.0448 | 0.7324 ± 0.0305 | 0.7472 ± 0.0199 | 0.8043 ± 0.0097 |
| KIBA | cold_target | 0.7231 ± 0.0888 | 0.5917 ± 0.0116 | 0.5187 ± 0.0212 | 0.6960 ± 0.0076 |

## Recommendation

KIBA carries the main results (2030 drugs, 117193 pairs; cold_drug test = 203-203 distinct drugs, 11618-12481 pairs), so random, cold_drug and cold_target are all statistically usable there. Davis is the secondary comparison: random and cold_target test sets are usable in pair count, but Davis cold_drug tests only 7-7 distinct drugs per seed (2653-2653 pairs that are highly correlated within a drug), so its cold_drug numbers describe generalisation to a handful of chemotypes and should be reported only with the seed spread, not as a headline. Davis cold_target likewise tests on only 38 kinases per seed and KIBA cold_target on 23 proteins per seed (pair counts are large but the effective sample for protein generalisation is the number of test proteins; KIBA cold_target test size also varies across seeds, see split table).
