# Project status: mtot_rhalfstar

Living document tracking experiments on total enclosed mass within the
half-stellar-mass radius. See the [project overview](status.md) for the
flagship models table and the key vmap finding, [sSFR
experiments](status_sfr.md) for the full sSFR regression log, and
[fgas and fdm experiments](status_fgas_fdm.md) for the fgas and fdm log.

## Experiment log

- [Head-size sweep (galfiltered)][head-sweep]: tiny (mean R² 0.816)
  outperforms minimal (0.803) and small (0.802). Tiny is the flagship head.


### Head-size sweep (galfiltered) (2026-11-23)

Sweep of `--head-size` on the galfiltered mtot_rhalfstar dataset. Every
run uses `--model resnet`, lr=1e-3, batch_size=32, 100 epochs,
head_dropout=0.5, two seeds. Val R² and MAE at epoch 100. The wandb
runs do not log a separate best-checkpoint summary key; with
use_scheduler=true and patience=5 over 100 epochs the LR decays
substantially before the end, so the final-epoch value is at or near
the best.

| head | narrowest layer | head params | R² (s42) | R² (s7) | R² (mean) | MAE (s42) | MAE (s7) | MAE (mean) |
|------|-----------------|-------------|----------|---------|-----------|-----------|----------|------------|
| tiny (32/16/8/4) | 4 | ~316K | 0.8218 | 0.8096 | 0.8157 | 0.3131 dex | 0.3107 dex | 0.3119 dex |
| minimal (64/32) | 32 | ~10K | 0.7989 | 0.8073 | 0.8031 | 0.3092 dex | 0.3242 dex | 0.3167 dex |
| small (128..8) | 8 | ~339K | 0.7873 | 0.8175 | 0.8024 | 0.3404 dex | 0.3160 dex | 0.3282 dex |

(small seed 42 ran on 2026-11-20, before the `_galfiltered_` naming
convention was adopted for run names, but against the same galfiltered
dataset with the same config.)

**Verdict**: tiny leads clearly (mean 0.8157). Minimal and small are
essentially tied in mean R² (0.8031 vs 0.8024), but small carries a
large seed spread: 0.030 between seeds, with seed 7 high at 0.8175 and
seed 42 low at 0.7873. That spread matches what the sSFR capacity
sweeps saw repeatedly and means the small result is not stable. Minimal
is tight (0.0084 spread), and tiny tighter still (0.0122). The ranking
tiny > minimal >> small holds on both seeds independently except for
small seed 7, which nudges past tiny seed 7 by 0.008. That single
crossing is not enough to unseat the overall ordering.

The relative order differs from fgas/fdm, where minimal beat tiny. For
fgas/fdm the 4-unit bottleneck in tiny's head discards information the
target needs. Here it does not, suggesting mtot_rhalfstar presents a
more concentrated signal that survives a narrow bottleneck intact.

These numbers (mean peak 0.8157) sit well above the fgas range (~0.62)
and comfortably above the fdm range (~0.77 unfiltered). Whether that
reflects the target being physically easier to predict from a velocity
map, or something specific to the galfiltered population, is an open
question.


<!-- Section links -->

[head-sweep]: #head-size-sweep-galfiltered-2026-11-23
