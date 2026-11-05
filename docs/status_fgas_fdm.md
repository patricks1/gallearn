# Project status: fgas and fdm

Living document tracking experiments on gas fraction and dark-matter fraction. See the [project overview](status.md) for the flagship models table and the key vmap finding, and [sSFR experiments](status_sfr.md) for the full sSFR regression log.

## Experiment log

Everything tried so far, newest first:

- [Galfiltered training: first runs][galfiltered-runs]: cross-evaluation of unfiltered models on the galfiltered split, then galfiltered training results for fgas and fdm; several gaps remain.
- [SV-logit scaling for fgas: abandoned][svlogit-fgas]: exact zeros map to logit ≈ -10, far from the bulk, causing metric crashes; and in raw fgas units its best checkpoint underperforms plain_std, so the scaled-space R² peak was an artifact.
- [fdm clamped predictions: switched to logit scaling][fdm-logit]: plain standardization clamps fdm predictions to ~0.8-0.96; logit resolves the near-1 tail.
- [A failed ablation, twice][ablation-fail]: two monkeypatch attempts silently did nothing and looked like clean null results.
- [Velocity-map ablation][vmap-ablation]: zeroing the vmap takes both targets to R² ≈ 0. The photometry carries nothing on its own. See [status.md][vmap-status] for the full writeup.
- [Pre-head dropout, on against off][head-dropout]: no reproducible effect, the clean test the dropout post-mortem never ran.
- [Epoch budget][epoch-budget]: 50 epochs was enough. Of twelve runs extended to 75, one improved, and it was the most constrained head.
- [Head-size sweep on the new targets][head-sweep-new]: a ~10K parameter head matches or beats the ~2.99M one on both targets. Bottleneck width matters, parameter count barely does.
- [Ceiling check: fgas and fdm][ceiling-new]: mass and size alone reach val R² 0.354 on fdm, a far higher bar than sSFR's 0-0.04.


### Galfiltered training: first runs (2026-10)

Before retraining, we ran the existing unfiltered models against the galfiltered split as a diagnostic. A cross-evaluation takes an unfiltered-trained model and scores it, with no retraining, on the galfiltered val set; it measures how far the unfiltered models transfer to the filtered population. All plain standardization.

| target | head | val R² (s42) | val R² (s7) |
|--------|------|-------------|------------|
| fdm | minimal | 0.029 | -0.086 |
| fdm | small | — | 0.009 |
| fgas | minimal | 0.316 | 0.387 |

The fdm models collapse to near-zero or negative $R^2$ on the galfiltered population because their predictions clamp to roughly 0.8-0.96 (median ~0.87) and never resolve the true near-1 tail, which reaches 0.998 — the clamping problem the [logit transform fix][fdm-logit] addresses. The fdm scatter plot (run on `fdm_head-small_seed7`) gave the first clear view of this. The fgas models keep moderate signal but fall well below their unfiltered val baseline (0.622/0.615).

I originally cross-evaluated the small fdm head because at seed 7 it had the lowest unfiltered-val $R^2$ (77.3%) in the head-size sweep. That choice was immaterial: the gap to the minimal head's 76.7% mean is a rounding error against 339k params (small) vs 10k (minimal), and the small head's own mean of 76.6% is actually *below* minimal. I have now cross-evaluated the minimal fdm head for both seeds (above). Both fdm and fgas cross-evaluation scatter plots survive as `eval_regressor_galfiltered-split.pdf` alongside their respective checkpoints.

We ran these first galfiltered-population training runs over roughly the same period as the fdm logit and svlogit experiments below. The training tables below show galfiltered training evaluated on the galfiltered val split, distinct from the cross-evaluation table above. For avg_sfr galfiltered results see [status_sfr.md][avgsfr-galfiltered].

**fgas.** These runs use the galfiltered fdm split against the galfiltered fgas HDF5. All plain standardization.

R² is in scaled space; it equals raw-space R² here because plain standardization is linear. MAE is in raw fgas units (intuitive space), from evaluate.py.

| head | R² (s42) | R² (s7) | R² (mean) | MAE (s42) | MAE (s7) | MAE (mean) |
|------|----------|---------|-----------|-----------|----------|------------|
| minimal | 0.520 | 0.509 | 0.515 | 0.0175 | 0.0176 | 0.0175 |
| small | 0.515 | 0.522 | 0.519 | 0.0174 | 0.0172 | 0.0173 |
| medium | 0.513 | 0.526 | 0.520 | 0.0180 | 0.0177 | 0.0179 |

The galfiltered R² (~0.51-0.52 across heads) sits roughly 10 percentage points below the unfiltered fgas R² (0.622/0.615). As with fdm, that gap is most likely a variance artifact rather than worse predictions, since galfiltering plausibly narrows the fgas target range and so shrinks the Var(y) denominator of R². Raw MAE is the fairer cross-population comparison. The svlogit galfiltered run (seed 7) reached a scaled-space R² of 0.598 at epoch 19, which looks like it beats plain_std but does not. That 0.598 is in sv_logit_std space, the same logit inflation that makes fdm's scaled R² misleading. Evaluated in raw fgas units the same checkpoint scores R² 0.448 and MAE 0.0188, both worse than every plain_std head (0.515-0.520 R², 0.017-0.018 MAE). The svlogit peak was a scaling artifact, not a real gain, quite apart from the metric crashes that led to [abandoning it][svlogit-fgas]. The head sweep is essentially flat. All three heads land at mean R² 0.515-0.520 with raw MAE ~0.017-0.018 gas-fraction units, small and medium edging minimal by less than the seed spread. Head size barely matters for galfiltered fgas.

**fdm.** Both plain_std runs peaked at epoch 3-5 with clamped predictions, the problem the [logit transform][fdm-logit] fixes, and degraded from there. The scaled-space and raw-space R² columns are the same statistic computed in different spaces, not a rescaling of one number: the scaled-space columns score each run in its own training space (log-odds for logit_std, standardized fdm for plain_std), while the raw-space columns score it in raw fdm units via evaluate.py, which also produces the raw-unit MAE. Only the raw-space columns compare across scalings. For plain_std the two agree, since standardization is a linear transform and R² is invariant to it.

Columns: scaled-space R² (sc.R²) and raw-space R² each per seed with mean, then raw-unit MAE per seed with mean. Means are shown only where both seeds exist.

| head | scaling | sc.R² (s42) | sc.R² (s7) | sc.R² (mean) | raw R² (s42) | raw R² (s7) | raw R² (mean) | MAE (s42) | MAE (s7) | MAE (mean) |
|------|---------|-------------|------------|--------------|--------------|-------------|---------------|-----------|-----------|------------|
| minimal | logit_std | 0.461 | 0.448 | 0.455 | 0.233 | 0.147 | 0.190 | 0.0358 | 0.0355 | 0.0356 |
| small | logit_std | 0.445 | 0.479 | 0.462 | 0.233 | 0.240 | 0.236 | 0.0355 | 0.0357 | 0.0356 |
| medium | logit_std | 0.438 | 0.473 | 0.456 | 0.217 | 0.221 | 0.219 | 0.0373 | 0.0349 | 0.0361 |
| minimal | plain_std | — | 0.273 | — | — | 0.273 | — | — | 0.0388 | — |
| small | plain_std | — | 0.272 | — | — | 0.272 | — | — | 0.0370 | — |

Two of the logit_std runs show the distortion plainly, nearly tied in log-odds space (0.448 vs 0.445) yet split widely in raw space (0.147 vs 0.233). That gap is the logit transform's doing: the sigmoid inversion expands fdm's packed near-1 tail, so a small log-odds difference on the extreme galaxies maps to a large raw-space one. R² in log-odds space is a poor proxy for R² in raw space, and "best checkpoint by logit val loss" need not be the best by raw-space R².

The raw-space R² values look alarmingly low next to the unfiltered head-size sweep's 0.763/0.769, but that comparison is invalid. R² normalizes by the target's variance, and galfiltering collapses the fdm spread by removing the entire low-fdm tail. The unfiltered population runs 0.001 to 0.999 with std 0.27, while the galfiltered val set runs 0.53 to 0.998 with std 0.076, a roughly 13x smaller variance (0.0057 vs 0.073). Since R² = 1 - RMSE²/Var(y), that shrunken denominator drives R² down even when predictions are accurate. Backing the raw-space R² out into raw fdm units, the galfiltered logit models reach RMSE about 0.066-0.070, roughly half the unfiltered model's 0.131. In absolute fdm units the galfiltered models predict more accurately, not less; the low R² reflects the narrower target range alone. Compare fdm models across the filtered and unfiltered populations by raw RMSE or MAE, not R². The logit_std sweep now covers minimal, small, and medium at both seeds. By raw R² the order is small (0.236) > medium (0.219) > minimal (0.190), but minimal's mean is dragged down by a wide seed spread (0.233 at seed 42 versus 0.147 at seed 7) while small and medium stay tight across seeds. By raw MAE minimal and small tie (0.0356) with medium marginally worse (0.0361). Small is the most consistent performer across both metrics and both seeds, echoing the galfiltered avg_sfr result that the small head suits the filtered population.

### SV-logit scaling for fgas: abandoned (2026-10-07)

The fdm logit transform ([see below][fdm-logit]) motivated trying the same approach for fgas. Gas fraction is also bounded in [0, 1], so logit similarly linearizes gradient signals near the edges. The obstacle is fgas's ~1.5% gas-free galaxies: their exact-zero values are a real physical state, and logit(0) = -inf.

We used the Smithson-Verkuilen squish to bring exact zeros inside the logit domain without an arbitrary epsilon:

    squish(x, n) = (x * (n - 1) + 0.5) / n

With n equal to the dataset image count (~12,650 for galfiltered with 11 projections), squish maps 0 to ~4e-5. logit(4e-5) ≈ -10, roughly 7 units below the bulk of fgas values in logit space (logit(0.06) ≈ -2.75). That gap dominated MSE loss and produced periodic metric crashes in val metrics.

The root cause is structural. The SV squish was designed for rating scales where every respondent is expected to land somewhere interior; it maps the extreme rating to a near-boundary value, not a genuine boundary event. Applied to a physical quantity with a real mass point at exactly zero, it pushes the zero stratum to a near-boundary position the formula never intended to handle. No squish parameter resolves this: smaller n pulls gas-free galaxies further from -inf but compresses the whole (0, 1) interval and reduces resolution everywhere else.

Reverted to plain standardization. The galfiltered fgas distribution is narrow (median 0.06, max ~0.20) with no structural pile-up near either edge, so plain standardization is appropriate and the exact zeros are well-behaved under it.

A later check confirmed that reverting to plain standardization gave up no real performance. The best svlogit checkpoint (seed 7, epoch 19) posts a scaled-space R² of 0.598, which once looked like it beat plain_std. That number is in sv_logit_std space, though, not comparable to plain_std's. Evaluated in raw fgas units the same checkpoint scores R² 0.448 and MAE 0.0188, both worse than every galfiltered plain_std head (R² 0.515-0.520, MAE 0.017-0.018). So even at its pre-crash peak, svlogit offered no real gain over plain standardization. The apparent 0.598 advantage was a scaling artifact of the same kind that inflates fdm's logit-space R².

### fdm clamped predictions: switched to logit scaling (2026-10-06)

Early scatter plots for fdm under plain standardization showed predictions clamped to roughly 0.8-0.96. The model compressed most of the distribution into a narrow band rather than resolving the near-1 tail. The cause is the distribution's shape: the galfiltered fdm dataset is unimodal and concentrated near 1 (range 0.33-0.999, median 0.90). In linear space, fdm=0.96 and fdm=0.99 differ by only 0.03, so MSE gradient on the dense near-1 tail is negligible. The optimizer regresses to the bulk rather than fitting the tail.

Switched `FdmTarget` to logit + standardize. Logit spreads the near-1 interval dramatically: logit(0.96) ≈ 3.18, logit(0.999) ≈ 6.91, roughly a 48x expansion of the key range. The optimizer now has a well-resolved gradient signal throughout the distribution. Predictions unscale through sigmoid, guaranteeing outputs in (0, 1).

No full re-sweep was run after this change, so the [head-size sweep][head-sweep-new]'s fdm numbers (val R² 0.763/0.769) are pre-logit results under plain standardization. Training metrics during a logit-transform run are in log-odds units and are not directly comparable to those baselines without unscaling first.

### A failed ablation, twice (2026-09-08)

The vmap ablation took three attempts, and the two failures are worth recording because both produced plausible-looking output.

The first patched `preprocessing.sasinh_imgs_sscale_vmaps`. Training never calls it. `LazyGalaxyDataset.__getitem__` reimplements those transforms inline against precomputed stats, and `cnn.ResNet.scaling_function` is vestigial during training. The second patched `__getitem__` itself, in the parent process. `train.main()` builds its DataLoaders with `num_workers=4`, and macOS spawns workers, so each one re-imported the module and got the unpatched class.

Both produced ablation runs bit-identical to their controls, which reads as a clean null result. What exposed it was the deltas being exactly 0.0000 across two independently seeded pairs. Genuinely null effects scatter around zero; exact agreement to four decimals means the same computation ran twice.

The fix was to stop patching. `zero_channels` now lives on `LazyGalaxyDataset` as an instance attribute, which reaches workers because the DataLoader pickles the dataset, and `train.main()` records it in `train_config` so a run's own config states which channels it trained on. The verification that should have run from the start reads batches through a `num_workers=4` DataLoader and checks the channel is zero there, not in the parent.

The general lesson for one-off experiment scripts here: verifying that a patch is installed is not the same as verifying it changes what the model sees.

### Pre-head dropout, on against off (2026-09-08)

`cnn.ResNet.forward` applied p=0.5 dropout to the pooled feature vector, a rate that arrived by convention and had never been tested here. The [dropout post-mortem][pm-dropout] tested only eval-time behavior, with both arms training at 0.5, and its no-dropout comparison point was StandardNet, a different architecture. Making the rate a parameter allowed the clean test. fgas, two head widths, two seeds:

| head | seed | p=0.5 | p=0.0 | change |
|---|---|---|---|---|
| small | 42 | 0.6150 | 0.5963 | -0.0187 |
| small | 7 | 0.6110 | 0.6319 | +0.0209 |
| small | mean | | | +0.0011 |
| minimal | 42 | 0.6228 | 0.6224 | -0.0004 |
| minimal | 7 | 0.6154 | 0.6034 | -0.0120 |
| minimal | mean | | | -0.0062 |

**Verdict**: no reproducible effect either way. Both mean changes sit inside the seed spread, and small's two seeds disagree in sign by almost equal amounts. This confirms the post-mortem's conclusion with better evidence than it had, and it means `head_dropout` can go to 0.0 to drop a hyperparameter nobody needs to justify, or stay at 0.5, with the data indifferent.

Worth recording how close this came to entering the doc as a finding. The seed-42 small run alone showed dropout helping by 0.019, with a tidy overfitting story attached, since removing dropout dropped train loss from 0.167 to 0.050. Seed 7 reversed it. That is the third time this project has caught a single-seed result that looked mechanistic and was not.

### Epoch budget: 50 was enough (2026-09-08)

The sweep ran at 50 epochs, and its first runs peaked at 36 to 40, close enough to the cap to raise the question. Rather than rerun, every run resumed from its own `checkpoint_epoch050.pt` for another 25 epochs. `train.py` saves a final checkpoint alongside best-loss ones, and a resumed run restores the optimizer state and the shuffling generator before running `range(start_epoch, start_epoch + n_epochs)` in the same wandb run. Two loop-state pieces do not survive the resume and reset instead: the `ReduceLROnPlateau` bad-epoch counter and the best-val tracker that gates checkpoint saves. The decayed learning rate itself carries over inside the optimizer state, and across a single 25-epoch resume with `patience=5` the counter reset only pushes any further LR decay a few epochs later, so a resume stays close enough to an uninterrupted 75-epoch run for this comparison.

Of twelve runs, exactly one improved. `fdm_head-tiny_seed7` moved from 0.7398 to 0.7618 with its best at epoch 60, then plateaued through 75. Every other run kept its pre-50 best, most by a wide margin.

The one that moved is the informative one. tiny is the most constrained head tested, and the [capacity work on sSFR][head-sweep-sfr] already found that smaller models fit more slowly and peak later. Without the extension, tiny's fdm result would have read 0.019 worse than it is, exaggerating the bottleneck effect the sweep above reports.

### Head-size sweep on the new targets (2026-09-08)

The first use of the new `--head-size` flag, which replaced the hardcoded head. Every run uses `--model resnet`, the avg_sfr split against each target's own dataset, lr=1e-3, batch_size=32, two seeds, 50 epochs then extended to 75 (see [epoch budget][epoch-budget]). Val R2 at each run's best checkpoint:

| target | head | narrowest layer | head params | seed 42 | seed 7 | mean |
|---|---|---|---|---|---|---|
| fgas | minimal (64/32) | 32 | ~10K | 0.6228 | 0.6154 | 0.6191 |
| fgas | tiny (32/16/8/4) | 4 | ~316K | 0.6163 | 0.6187 | 0.6175 |
| fgas | small (128..8) | 8 | ~339K | 0.6150 | 0.6110 | 0.6130 |
| fgas | large (2048..64) | 64 | ~2.99M | 0.6015 | 0.5701 | 0.5858 |
| fdm | minimal (64/32) | 32 | ~10K | 0.7631 | 0.7694 | 0.7663 |
| fdm | tiny (32/16/8/4) | 4 | ~316K | 0.7331 | 0.7618 | 0.7475 |
| fdm | small (128..8) | 8 | ~339K | 0.7585 | 0.7728 | 0.7657 |
| fdm | large (2048..64) | 64 | ~2.99M | 0.7523 | 0.7317 | 0.7420 |

(fgas large seed 42 is `lively-morning-94`, reused rather than rerun. It ran 100 epochs and peaked at 40, inside this sweep's budget, so its best checkpoint is comparable.)

**Verdict**: cutting the head from ~2.99M parameters to ~10K costs nothing and helps slightly. The large head posts the lowest mean on both targets, and comes last in three of the four target-seed combinations, losing only to fdm's tiny at seed 42. That is the one ordering here with real support. Among the three narrow heads on fgas the spread is 0.006, smaller than some within-config seed spreads, so they are indistinguishable there.

Parameter count is not the operative variable. On fdm, tiny sits 0.019 below minimal while carrying 32 times more parameters. What separates them is the narrowest layer, 4 units against 32. A 4-dimensional bottleneck discards real information for a target this well determined, which fgas at 0.62 tolerates and fdm at 0.77 does not. The lesson is to avoid pinching the head, not to shrink it.

Even that reading deserves caution. tiny's gap to minimal on fdm swings from 0.030 at seed 42 to 0.008 at seed 7, and small's own seed spread is 0.014, so the direction holds on both seeds while the magnitude does not. This doc has declined to call weaker versions of that pattern a real effect before.

These runs also show much tighter seed variance than the sSFR sweeps did, roughly 0.004 against the 0.03 to 0.05 that flipped conclusions there. Higher-signal targets make the two-seed rule cheaper to satisfy, though the [dropout][head-dropout] result below shows it is still doing necessary work.

### Ceiling check: fgas and fdm (2026-09-08)

The same scalar-feature baseline the sSFR project ran, refit for the new targets: linear regression and gradient-boosted trees on `log10(Mstar)` and `Re`, no images, against the same train/val galaxies the CNN runs use, with the target standardized on train rows alone.

| target | features | model | train R2 | val R2 |
|---|---|---|---|---|
| fgas | log(Mstar) | linear | 0.0280 | 0.0048 |
| fgas | log(Mstar) | GBR | 0.5268 | -0.1787 |
| fgas | log(Mstar) + Re | linear | 0.0372 | 0.0144 |
| fgas | log(Mstar) + Re | GBR | 0.5473 | 0.0563 |
| fdm | log(Mstar) | linear | 0.1062 | 0.1036 |
| fdm | log(Mstar) | GBR | 0.4637 | 0.1238 |
| fdm | log(Mstar) + Re | linear | 0.1226 | 0.1295 |
| fdm | log(Mstar) + Re | GBR | 0.5920 | 0.3539 |

Running this first turned out to matter for fdm. Mass and size alone reach val R2 0.354 there, unlike sSFR, where the same baseline got 0.00 to 0.04 and made almost any CNN number look like real signal. A dark-matter-fraction model has to clear 0.354 before it demonstrates anything beyond a two-variable fit, which is also what makes the [vmap ablation][vmap-ablation]'s 0.00 so damning: without that channel the CNN cannot even match the trivial baseline.

Rows here are per image rather than per galaxy, so one galaxy appears about eleven times with identical scalars and different orientations. Val galaxies stay disjoint from train galaxies, so the val numbers remain honest.


<!-- Section links -->

[galfiltered-runs]: #galfiltered-training-first-runs-2026-10
[svlogit-fgas]: #sv-logit-scaling-for-fgas-abandoned-2026-10-07
[fdm-logit]: #fdm-clamped-predictions-switched-to-logit-scaling-2026-10-06
[ablation-fail]: #a-failed-ablation-twice-2026-09-08
[head-dropout]: #pre-head-dropout-on-against-off-2026-09-08
[epoch-budget]: #epoch-budget-50-was-enough-2026-09-08
[head-sweep-new]: #head-size-sweep-on-the-new-targets-2026-09-08
[ceiling-new]: #ceiling-check-fgas-and-fdm-2026-09-08
[vmap-ablation]: status.md#project-status-the-vmap-carries-the-signal-2026-09-08
[vmap-status]: status.md#project-status-the-vmap-carries-the-signal-2026-09-08
[backbone-dropout]: status_sfr.md#dropout-placement-dropout-in-the-backbone-2026-07-20
[pm-dropout]: status_sfr.md#post-mortem-dropout
[head-sweep-sfr]: status_sfr.md#capacity-sweep-head-width-2026-07-19
[considered]: status_sfr.md#considered-and-set-aside-a-different-architecture
[avgsfr-galfiltered]: status_sfr.md#galfiltered-training-avg_sfr-2026-10
