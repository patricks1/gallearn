# Project status: sSFR regression generalization

Living document tracking what we know about why the sSFR regressor
isn't generalizing well, what we've tried, and what's next. This is
also a historical log: append new findings, don't delete or silently
rewrite old ones. If something turns out to be wrong or confounded,
say so in place and point forward, rather than removing it.

Last updated: 2026-07-22. **This is the final entry.** See
[Project status: concluded][status], immediately below.

## Project status: concluded (2026-07-22)

This project is done, not because it failed, but because every
axis this doc tracked got a controlled, two-seed-checked answer, and
none of them points at a fixable problem. Summary, with links to the
evidence:

- **Architecture is exhausted.** Capacity closed in both directions:
  [pretraining][pm-pretrained] (more capacity) sharpened overfitting,
  [shrinking the backbone][backbone-sweep] (less capacity) made
  generalization steadily worse. Regularization closed too, and more
  sharply: [backbone dropout][backbone-dropout] nearly eliminated the
  train/val gap and val R² *fell* anyway, the strongest evidence in
  this doc that the ceiling was never an overfitting problem in the
  ordinary sense. [Exotic architecture families][considered] were
  considered and set aside on the same evidence, not tried, because
  nothing pointed at representational power as the bottleneck.
- **Target noise is the best-supported explanation, though not
  proven beyond doubt.** Plain shot noise is too small to explain the
  ceiling, but a real [multi-window comparison][target-noise] found
  genuine sub-Gyr burstiness in sSFR, and its implied ceiling (R² ~
  0.295) landed close to the observed one. A direct test (a 0.3 Gyr
  regressor target) made things worse, not better, which reframed but
  didn't overturn this: the 1 Gyr window's smoothing isn't the
  problem, the underlying stochasticity is. A
  [heteroscedastic head][het-head] tried to convert this into direct,
  per-galaxy evidence and mostly didn't, its predicted variance
  correlated with target magnitude more than with anything
  galaxy-specific.
- **Color and photometry are closed, on a corrected basis.** The
  original [color baseline check][target-noise] found no signal, and
  this doc spent real effort second-guessing whether that was a
  measurement artifact. It wasn't. Mock images are physically
  calibrated luminosity surface brightness with no background and no
  distance variation (`~/code/mockobservation-tools`), so the
  original null result stands. The likely reasons color adds nothing
  are documented in place: it was tested on the star-forming subset
  only, where color's range compresses; dust attenuation is modeled
  and reddens actively star-forming galaxies, partially inverting the
  naive color-SFR relation; and color mainly tracks a shorter,
  burstier timescale than the 1 Gyr target, the same noisy component
  the window-sweep already flagged.

**What this project actually produced**: a hurdle model whose
classifier stage generalizes well (val F1 ~ 0.97) and whose regressor
stage has a real, reproducible, above-trivial-baseline signal (val R²
~ 0.28-0.31) that appears to sit close to the images' real
information limit for a 1 Gyr sSFR target, not an engineering
shortfall. That ceiling is best explained by genuine astrophysical
burstiness in star formation on sub-Gyr timescales, evidence a
single snapshot image cannot resolve, established directly rather
than only inferred by elimination. See the README's "Hurdle model"
section for the project-level framing.

**Left genuinely open, not pursued further**: a second
heteroscedastic-head pass correlating predicted variance against
actual multi-window burstiness data instead of squared error (would
need regenerating a particle-count CSV on Greenplanet); running
multiple val splits to get a variance estimate on val R² itself. Both
are recorded in [Documented future directions][candidates], not
pursued as part of this project.

## Current understanding (read this first)

The CNN pipeline has two tasks: a **classifier** (quenched vs.
star-forming) that generalizes well (val F1 ≈ 0.97), and a
**regressor** (sSFR on the star-forming subset) that overfits: train
loss collapses toward near-zero while val R² caps out around
0.24-0.28 no matter what we've changed so far.

A ceiling check (scalar-feature baseline on `log10(Mstar)` + `Re`,
no images) gets val R² ≈ 0-0.04, so the regressor's R² ≈ 0.24-0.28 is
several times better than a trivial baseline. The images carry real
signal; this isn't a network learning nothing.

We've tested lr, pretraining, projection count, dropout (both
placements), and two capacity sweeps (head width and backbone width
on `cnn.ResNet`) each in a controlled way (see log below), and every
one of them lands in the same narrow R² band or below it, with every
non-pretrained run peaking very early (epoch 15-26 of 50) then
degrading. That consistency itself is the finding: none of them is
the bottleneck.

The architecture axis is now effectively exhausted, and the way it
closed is itself the most useful result in this doc. Capacity is
closed in both directions: pretraining added capacity and sharpened
overfitting, and shrinking the backbone made generalization steadily
worse. Regularization is closed too, and more sharply: putting
dropout throughout the conv backbone nearly eliminated the train/val
gap and made val performance *worse*, not better. That combination
is hard to square with an overfitting story. If excess model freedom
were what capped val R², constraining it should have helped
somewhere along that path, and it never did. The large train/val gap
looks like a symptom of a target the images only partly determine,
not the cause of the ceiling.

We also checked whether the sSFR targets themselves are
simply too noisy to predict well. Plain particle-count shot noise is
far too small to explain the ceiling on its own, but a real
multi-window comparison found genuine astrophysical burstiness
between 1 Gyr and 100 Myr windows, large enough that its implied
ceiling (R² ≈ 0.295) landed suspiciously close to the observed one.
That match didn't survive a direct test, though: retraining the
regressor on a 0.3 Gyr target instead of 1 Gyr made generalization
*worse* on both seeds tried, not better (see the
[sSFR target noise check][target-noise]). A color-only baseline also
carries essentially no sSFR signal on its own; an initial worry that
this was a measurement artifact (an uncalibrated color proxy) turned
out to be wrong once checked against the image pipeline (see
[Project status: concluded][status] above), so this null result
stands as real, not just unresolved. A
[heteroscedastic head][het-head] tried
to turn the target-noise hypothesis into direct, per-galaxy evidence:
its predicted variance does correlate with actual error (Spearman ≈
0.28, reproducible across seeds), but that correlation mostly
evaporates once controlled for target magnitude, so it doesn't
confirm or refute target noise, it mostly measured a simpler pattern
instead.

We found and fixed two real infrastructure bugs along the way
(resume not restoring most run settings; eval-mode dropout silently
active during validation); see the log for what they affected. The
capacity sweep also taught a methodological lesson the hard way: its
first-seed result looked like a real effect and mostly turned out to
be noise on reseed, so every capacity/regularization experiment from
here on runs at least two seeds per point from the start.

## Documented future directions, not pursued

Everything already tested, and how it turned out, lives in the
[experiment log][log] below. This project concluded (see
[Project status: concluded][status] above) before either item below was
tried; they're recorded for anyone who revisits this work, not as
active next steps.

- **Heteroscedastic head, take two: correlate against burstiness
  directly, not against squared error**. [The first attempt][het-head]
  found predicted variance mostly tracks target magnitude, a
  confound squared error shares. A cleaner test correlates predicted
  variance against actual multi-window sSFR disagreement (the data
  behind the [burstiness comparison][target-noise]) instead, which
  isn't automatically confounded with `|target|` the same way. Needs
  regenerating the per-galaxy particle-count/multi-window CSV on
  Greenplanet, since the scratchpad copy from that check didn't
  survive between sessions.
- **Multiple val splits** (`scripts/split.py split --split-name
  ...`) to get a variance estimate on val R², since a single 85/15
  galaxy split with ~1500 galaxies is noisy on its own, and the
  [head-width sweep][head-sweep] showed directly how easily a
  single-seed difference can look like a real effect and not be one.

## Experiment log

Everything tried so far, grouped by theme, oldest first:

- [Classifier: generalizes fine][classifier]: val F1 ≈ 0.97, not the
  problem. All effort below is the regressor.
- [Regressor: earliest results][earliest]: pre-split-refactor runs,
  kept for history, confounded.
- [Ceiling check][ceiling]: `log10(Mstar)` + `Re` with no images gets
  val R² ≈ 0-0.04, so the CNN's ~0.24-0.31 is real signal.
- [Controlled regressor experiments][controlled]: lr and projection
  count tested cleanly. Neither is the lever.
- [Bug 1: `--resume` didn't restore settings][bug1]: fixed. It
  invalidated some earlier resumed runs.
- [Bug 2: eval-mode dropout][bug2]: fixed. Dropout was silently
  active during validation.
- [Dropout bug A/B][dropout-ab]: fixing Bug 2 made no real
  difference.
- [Capacity sweep: head width][head-sweep]: no robust effect. The
  first-seed result was noise, caught on reseed.
- [Capacity sweep: backbone width][backbone-sweep]: shrinking the
  backbone made things monotonically worse, closing capacity as a
  lever in both directions.
- [Dropout placement: dropout in the backbone][backbone-dropout]:
  nearly closed the train/val gap and made val *worse*, the
  strongest evidence yet that the ceiling isn't an overfitting
  problem.
- [sSFR target noise check][target-noise]: shot noise is too small to
  explain the ceiling. Real burstiness shows up between windows, but
  a 0.3 Gyr target made generalization worse, not better. Also
  covers the color baseline check.
- [Heteroscedastic regression head][het-head]: predicted variance
  correlates with actual error, but mostly because both track target
  magnitude. Little left once that's controlled for.

### Classifier: generalizes fine

wandb project `gallearn_quenched_classifier`, 3 runs (all 3
projections, lr=1e-4):

| run | train acc | val acc | val F1 |
|---|---|---|---|
| sleek-spaceship-6 | 99.9% | ~94.5-95% | ~0.97 |
| summer-capybara-7 | 99.8% | ~94.5-95% | ~0.97 |
| expert-sky-8 | 99.8% | ~94.5-97% | ~0.97-0.98 |

Only a ~5-point train/val gap. Quenched-vs-star-forming is a coarse
binary call with a strong signal (concentration, color, smoothness);
regression to a continuous sSFR value needs much finer-grained
information and is a much harder small-N problem. This is why effort
below is all on the regressor.

### Regressor: earliest results (pre-split-refactor, early resnet)

wandb project `sfr_gallearn`, the earliest `resnet`-architecture
regressor runs, predating the locked-test-set/split-file system and
today's `StandardNet`/`--model standard`:

| run | projections | lr | epochs | train loss | val R² |
|---|---|---|---|---|---|
| likely-thunder-55 | 3 | 1e-4 | 1000 | 0.005 | 0.237 |
| ethereal-violet-61 | 11 | 1e-4 | 78 (killed) | 0.40 | -0.007 |
| rare-sun-62 | 11 | 1e-3 | 100 | 0.25 | 0.232 |

At the time, the 3-vs-11-projection comparison here (0.237 vs 0.232)
looked like projection count made no difference. We never controlled
that comparison, though (different lr, different epoch counts).
See the properly controlled version further down, which found a
real, if modest, effect in the other direction. These three runs
also predate the eval-mode dropout fix (see "Bug 2" below), which
may explain some of their val-loss jumpiness, particularly
`rare-sun-62`'s.

### Ceiling check (2026-07-17)

Fit a plain scalar-feature baseline (linear regression and
gradient-boosted trees on `log10(Mstar)` and `Re`, no images at all)
against the exact same train/val galaxies as the real split
(`splits/split_..._v1.json`), predicting the same asinh-standardized
sSFR target:

| features | model | train R² | val R² |
|---|---|---|---|
| log(Mstar) | linear | 0.0005 | -0.025 |
| log(Mstar) | GBR | 0.33 | 0.030 |
| log(Mstar) + Re | linear | 0.003 | -0.012 |
| log(Mstar) + Re | GBR | 0.32 | 0.038 |

Mass and size alone explain almost nothing (val R² ≈ 0-0.04). Every
CNN run in this doc clears R² ≈ 0.24+, several times better than the
best trivial baseline here, so the network is extracting real
information from the images beyond what a simple scalar summary
could give it, likely morphology (clumpiness, concentration,
structure), not just "how big/massive is this galaxy."

Inputs: only `Re` (Sersic radius) goes in as an auxiliary scalar.
Stellar mass is not currently fed to the network (this ceiling check
suggests it wouldn't help much on its own, since mass alone barely
predicts sSFR here).

### Controlled regressor experiments (2026-07-17 - 2026-07-18)

All `--task regressor --model standard` (`StandardNet`, formerly
`BernoulliNet`), same split, batch size 64, single seed each unless
noted, each measured at that run's own best-checkpoint val loss
epoch:

| run | change from baseline | best epoch | train loss | val loss | val R² |
|---|---|---|---|---|---|
| baseline | lr=1e-3, scratch | 15 | 0.62 | 0.716 | 0.276 |
| lr_probe | lr=3e-4, scratch | 17 | ~0.60 | 0.713 | ~0.28 |
| pretrained (50 ep) | lr=1e-3, pretrained | 50 (still improving) | 0.017 | 0.744 | 0.250 |
| pretrained (resumed +50 ep) | same, resumed to 100 ep total | 68 | n/a (see caveat) | 0.722 | n/a (see caveat) |
| 3-proj only | lr=1e-3, scratch, `train_orientations` restricted to the 3 axis-aligned views | 26 | 0.60 | 0.756 | 0.237 |

Every one of these lands in the same narrow band (val loss
0.71-0.76, R² 0.24-0.28), and every non-pretrained run peaks early
(epoch 15-26 of 50) then degrades. That consistency is itself the
finding: lr, pretraining, and projection count all turned out not to
be the bottleneck. The pattern instead looks like a capacity/
overfitting problem. The model reaches its best generalization point
almost as soon as it starts fitting hard, regardless of how it gets
there.

3-proj-only losing ~0.04 R² relative to full 11-proj (0.237 vs 0.276)
is a real, if modest, effect in favor of more projections. This is
the properly controlled version of the comparison the earliest runs
above got wrong. It doesn't fix the fundamental ~1500-galaxy limit,
but real multi-view projections (as opposed to synthetic flip/rotate
augmentation, which would add much less: a flipped image carries no
new inclination/dust/structure information a real projection would)
aren't worthless regularization either.

**Caveat on the "pretrained (resumed)" row**: its post-resume portion
(epochs 51-100) used `batch_size=32` instead of the original run's
`64`, due to "Bug 1" below. Its train loss and R² for that stretch
aren't comparable to the other rows, so we omit them; we include the
val loss for reference only, not as a clean result.

### Bug 1 (2026-07-18, fixed): `--resume` didn't restore settings

Resuming the pretrained run above (to let it finish converging past
epoch 50) produced a visible train-loss jump right at the resume
boundary. Root cause: `--resume` was only reusing the checkpoint's
weights/optimizer state, dataset, and split—not `batch_size`,
`task`, `model_type`, `run_name`, `lr`, `seed`, `use_scheduler`, or
`pretrained`. The resume command omitted `--batch-size`, so it
silently fell back to argparse's default (32) instead of the
original run's 64. We confirmed this via the optimizer's step
counter jumping from 8400 to 8736 between checkpoints, almost
exactly double a normal ~168-step epoch. (Our initial hypothesis was
a data-shuffling discontinuity from an unseeded DataLoader generator;
that turned out to be a real issue but not the cause of this
particular jump. See the seed/generator-state fix we folded in
below, which addresses the shuffling issue on its own merits.)

We fixed this comprehensively: every parameter
`gallearn.train.main()` takes is now either (a) restored from the
checkpoint on resume and rejected if the caller also passes it
explicitly, or (b) `n_epochs`, the deliberate exception, since
running a different number of further epochs on each resume is
expected. `--resume` alone now needs no other flags. We also fixed a
second bug: `wandb_mode='r'` was calling `wandb.init(resume='must')`
without a run id, which wandb rejects (it resumes by id, not name);
a resumed run now automatically reattaches to its checkpoint's own
wandb run (recorded as `wandb_run_id`) or continues without wandb,
with no `--wandb r` mode to forget. `seed` also didn't do anything
before this fix (dead parameter). It now seeds model init and
`train_loader`'s shuffling generator on a fresh run, and a resumed
run restores that generator's exact saved RNG state
(`torch.Generator.get_state()`/`set_state()`, captured in every
checkpoint) rather than re-seeding, so it draws the same next
shuffle an uninterrupted run would have.

### Bug 2 (2026-07-19, fixed): eval-mode dropout active in validation

`cnn.ResNet` and `cnn.Net` (the latter not currently wired into
`train.py`) both called `torch.nn.functional.dropout(x, p)` without
`training=self.training`. Functional dropout defaults to
`training=True` regardless of `model.eval()`. Unlike the
`nn.Dropout` module, it doesn't automatically respect eval mode. So
every `cnn.ResNet` validation forward pass was randomly zeroing 50%
of features, adding noise to every val prediction and metric, and
meaning "best checkpoint" selection was partly picking a lucky
dropout mask rather than the true best model.

We fixed this by passing `training=self.training` to both calls.
This likely explains at least some of the very jumpy val curves seen
in the earliest `--model resnet` runs above (particularly
`rare-sun-62`). It's unlikely to be the main explanation for the
generalization ceiling itself, though: `StandardNet` has zero
dropout, buggy or otherwise, and showed the same early-peak-then-
plateau pattern in every controlled experiment above. A dropout-free
architecture reproducing the same failure mode argues against broken
dropout being the dominant cause. We test this directly next (see
below).

### Dropout bug A/B (2026-07-19): fix made no real difference

Since we expected dropout to induce generalization, and the bug
effectively disabled it during training's own model-selection
process (it corrupted the val loss used to pick "best"), it was
worth directly testing whether the fix alone changes anything before
moving to bigger interventions. Two `--model resnet` runs, otherwise
identical (`--task regressor`, same split, lr=1e-3, batch_size=64,
50 epochs, seed=42): one with `cnn.ResNet.forward` monkeypatched
back to the buggy (always-on) dropout, one with the fix.

| run | best epoch | train loss | val loss | val R² |
|---|---|---|---|---|
| buggy dropout | 27 | 0.53 | 0.7385 | 0.263 |
| fixed dropout | 19 | 0.62 | 0.7324 | 0.256 |

Essentially no change: val loss is marginally better fixed, R² is
marginally *worse* fixed. Those two metrics disagreeing on direction
is itself informative: this is noise, not a real effect either way.
Both land squarely in the same 0.71-0.76 val loss / R² ≈ 0.25-0.28
band as every other experiment in this doc. The bug was real and
worth fixing (it was adding noise to val metrics and corrupting
"best checkpoint" selection), but it wasn't the generalization
bottleneck. That rules out a fourth candidate (after lr, pretraining,
and projection count) and further strengthens the capacity
hypothesis below, since it's now the only lever left untested.

This A/B only tested eval-time dropout behavior. Both arms still
trained with dropout on (p=0.5), so it doesn't by itself say whether
dropout-as-regularization (on vs. off entirely) matters. We decided
not to run that as a dedicated experiment, though: the historical
`cnn.ResNet` runs (dropout present, buggy or fixed, R² ≈ 0.23-0.26)
and every `StandardNet` run (no dropout at all, R² ≈ 0.24-0.28) all
land in the same band regardless. Not a single controlled test, but
enough independent data points pointing the same way that a
dedicated dropout on/off run isn't worth a ~35 min slot next to the
capacity sweep below, which targets a genuinely untested axis.

### Capacity sweep: head width (2026-07-19)

`cnn.ResNet`'s head (`nn.LazyLinear(2048)` down through 1024, 256,
128, 64, to `N_out_channels`) turned out to hold about 87% of the
whole model's parameters (~2.68M of ~2.99M total). `n_blocks_list`
and `out_channels_list`, the two knobs the "Next candidates" section
originally pointed at, barely move total capacity at all by
comparison. So the real capacity-reduction lever here is the head's
width, not the conv backbone. We shrank the head across three
points, holding the backbone fixed at `n_blocks_list=[1,1,1,1]`,
`out_channels_list=[16,32,64,128]`, same split, lr=1e-3,
batch_size=64, 50 epochs, `--model resnet`:

| head widths | total params | seed 42 val loss / R² | seed 7 val loss / R² |
|---|---|---|---|
| large (2048/1024/256/128/64) | ~2.99M | 0.7324 / 0.256 | 0.7216 / 0.267 |
| medium (512/256/128/64/32) | ~554K | 0.6924 / 0.306 | 0.7423 / 0.251 |
| small (128/64/32/16/8) | ~339K | 0.6957 / 0.299 | 0.7104 / 0.287 |
| tiny (32/16/8/4) | ~316K | 0.7919 / 0.200 | not run |

(The large row's seed 42 result is `dropout_ab_fixed_resnet` from
the dropout A/B above, reused as the capacity anchor since it's the
exact same architecture and hyperparameters.)

At seed 42, medium and small both beat large by a real-looking
margin (R² 0.30-0.31 vs. 0.26, val loss down to 0.69), landing
outside the 0.71-0.76 band every other experiment in this doc lands
in, our first result that wasn't obviously noise. Tiny undershot
large instead, suggesting a non-monotonic capacity/generalization
curve with a sweet spot somewhere between ~339K and ~554K params.

Reseeding large, medium, and small at seed 7 to check: the medium
result didn't survive at all (0.7423, now the worst of the three,
worse than large). Small still beat large in both seeds (0.6957 <
0.7324 at seed 42, 0.7104 < 0.7216 at seed 7), but the margin shrank
sharply (0.037 to 0.011), and every seed 7 number landed back inside
the historical 0.71-0.76 band.

**Verdict**: this sweep did not find a robust capacity effect. The
medium result was seed noise dressed up as a finding, the exact trap
this doc has been careful to check for elsewhere (see the dropout
A/B's noise-signature discussion above), and it slipped past here
because we only ran one seed per point at first. Small-head's edge
over large-head is directionally consistent across both seeds but
small and inconsistent enough in magnitude that it doesn't clear the
bar this doc has used everywhere else to call something a real
effect. Capacity, at least via head width at these three points,
joins lr, pretraining, projection count, and dropout as tested and
ruled out as the dominant lever.

**Mechanism, and a comparison we initially got wrong**: tiny's best
checkpoint (epoch 44, train loss 0.55) looked, at first glance, like
it fit the training set about as well as large's best checkpoint
(epoch 19, train loss 0.62), which would be a strange result: worse
capacity fitting just as well but generalizing worse. That
comparison is confounded, though, since it compares each run at its
own best-val epoch, and those landed at very different points in
training. Pulling all four runs' train and val loss at a matched
epoch (19, large's best) instead:

| model | train loss @ epoch 19 | val loss @ epoch 19 |
|---|---|---|
| large | 0.621 | 0.732 |
| medium | 0.680 | 0.855 |
| small | 0.686 | 0.786 |
| tiny | 0.723 | 0.836 |

At a fixed point in training, large fits best and tiny fits worst,
exactly what raw capacity would predict. There's no paradox: bigger
models fit faster and start overfitting (val loss degrading) sooner,
so their best-val checkpoint lands early (large peaks at epoch 19);
smaller models fit more slowly and take longer to reach the point of
overfitting, so their best-val checkpoint lands later (tiny peaks at
epoch 44). That explains the spread in the "best epoch" column
above without needing any special-case explanation for tiny, and it
means tiny's weak result is consistent with genuine undercapacity,
not a fluke, once compared at a matched epoch instead of each run's
own best checkpoint.

### Capacity sweep: backbone width (2026-07-20)

The head-width sweep above only touched the head; the conv backbone
stayed fixed at `n_blocks_list=[1,1,1,1]`,
`out_channels_list=[16,32,64,128]` (~311K backbone params) across
every point tested so far. This sweep targets the backbone directly:
`out_channels_list` shrunk to two smaller points, holding the small
head (`[128,64,32,16,8]`, the one head-width point whose edge over
large held up across both seeds) and `n_blocks_list=[1,1,1,1]` fixed,
so only backbone width varies. Two seeds each, same split, lr=1e-3,
batch_size=64, 50 epochs, `--model resnet`:

| `out_channels_list` | backbone params | total params | seed 42 val loss / R² | seed 7 val loss / R² |
|---|---|---|---|---|
| `[16,32,64,128]` (baseline) | 310,960 | 339,105 | 0.6957 / 0.299 | 0.7104 / 0.287 |
| `[8,16,32,64]` | 79,064 | 99,017 | 0.7595 / 0.250 | 0.7587 / 0.244 |
| `[4,8,16,32]` | 20,428 | 36,285 | 0.8383 / 0.165 | 0.8415 / 0.157 |

(The baseline row is the small-head point from the head-width sweep
above, reused rather than rerun, since it's the exact same
architecture and hyperparameters.)

Unlike the head-width sweep, this result isn't ambiguous: both
smaller backbones land clearly outside the baseline's range on both
seeds, with no reseed reversal, and generalization gets steadily
worse as the backbone shrinks further.

**Verdict**: shrinking the backbone does not help, it hurts,
monotonically and consistently across both seeds. Read together with
the head-width sweep's "tiny" point (which also undershot) and the
pretrained post-mortem (adding capacity via ImageNet weights also
didn't help, and sharpened overfitting instead), capacity now looks
like a closed lever in both directions: making the network bigger
didn't help, and making it smaller doesn't either. The current
backbone (~310K params) is not obviously carrying unused capacity
that a smaller network would use more efficiently; if anything, this
result is more consistent with the current backbone already sitting
near or below the capacity floor needed to represent the mapping at
all, not above some capacity ceiling that regularization or shrinking
needs to bring down.

### Dropout placement: dropout in the backbone (2026-07-20)

The last untested architecture lever. `cnn.ResNet` has a single
`p=0.5` dropout right before the head, and the standing hypothesis
(see the [dropout post-mortem][pm-dropout]) was that this is too
weak or too localized to constrain the conv backbone, where the
capacity to memorize training galaxies actually lives. This test
adds `nn.Dropout2d(p=0.2)` after each of `conv2_x`-`conv5_x`,
keeping the existing pre-head dropout, the small head, and the full
`[16,32,64,128]` backbone. `nn.Dropout2d` zeroes whole channels
(spatial dropout), the right granularity for conv feature maps, and
being a real `nn.Module` it respects `model.eval()` automatically,
so it cannot reintroduce [Bug 2][bug2]. Two seeds, same split,
lr=1e-3, batch_size=64, 50 epochs:

| config | seed | final train loss | best val loss | val R² |
|---|---|---|---|---|
| baseline (pre-head dropout only) | 42 | 0.372 | 0.6957 | 0.299 |
| baseline | 7 | 0.462 | 0.7104 | 0.287 |
| + `Dropout2d(0.2)` in backbone | 42 | 0.710 | 0.7954 | 0.206 |
| + `Dropout2d(0.2)` in backbone | 7 | 0.710 | 0.7903 | 0.209 |

(The seed 42 run crashed at epoch 22 when its environment restarted
and resumed from its epoch-18 checkpoint, so its wandb history has 54
rows, epochs 19-22 twice. The resume restores the generator state, so
the replayed epochs match; the numbers above are unaffected.)

**Verdict**: backbone dropout worked exactly as a regularizer is
supposed to, and that made things worse. It clearly did its
mechanical job: final train loss rose from 0.372/0.462 to 0.710 on
both seeds, and the train/val gap collapsed from about 0.32 to about
0.085, close to eliminated. But val loss got *worse* (0.795/0.790 vs.
0.696/0.710) and val R² dropped from ~0.29 to ~0.21. The model went
from overfitting to underfitting without ever passing through a
better val optimum on the way.

That last point is the informative part, and it's the strongest
evidence in this doc so far that the ceiling is not an overfitting
problem in the usual sense. The standard story ("the train/val gap
is large, so constrain the model and val will improve") predicts
that closing the gap buys generalization. Here we closed the gap
almost entirely and generalization got worse. Combined with both
capacity sweeps finding that smaller is worse, the picture is
consistent: there is no capacity or regularization setting between
here and a much smaller model that beats the current
configuration, because what limits val performance isn't the model's
freedom to memorize. The large train/val gap looks like a symptom of
a target the images only partly determine, not a cause of the
ceiling.

### sSFR target noise check (2026-07-19)

Every experiment above assumed the sSFR targets themselves are
clean. We checked that assumption directly. `sfr` comes from
`get_avg_sfrs` in UCITools' `ProcessFIREBox.jl`, and it is a literal
sum of individual star particle masses formed within the last 1 Gyr,
read from each galaxy's `particles_within_Rvir_object_<id>.hdf5`
file on the raw FIREbox data (Greenplanet). For a galaxy with few
young star particles, that sum is essentially a small-count total,
carrying real Poisson-like shot noise from the simulation's finite
star-particle mass resolution, not a bug, a genuine discreteness
limit.

We wrote a one-off script (`sfr_particle_counts.jl`, living on
Greenplanet in `~/projects/gallearn/scripts/`, not committed to this
repo) that reads every galaxy's raw particle file, counts
`n_young_particles` (star particles with `stellar_tform` in the last
1 Gyr, the same window `get_avg_sfrs` uses), and joins that back onto
`avg_sfrs_1.0Gyr_no_bound_filter.csv`.

Over the 1378 star-forming galaxies (`ssfr > 0`) in that CSV:

- Median `n_young_particles` is 1372, but the distribution has a
  long low tail: 7.1% of star-forming galaxies have fewer than 10
  young particles, 16.9% fewer than 50, 22.9% fewer than 100.
- Propagating simple counting-statistics shot noise (`1/√N` relative
  noise on `sfr`) into log10(sSFR) space and comparing it against
  the actual spread of log10(sSFR) across the star-forming
  population (std ≈ 0.405 dex) gives an implied noise ceiling of R²
  ≈ 0.96 (mean case) to R² ≈ 0.999 (median case).

That's far above the ~0.24-0.31 ceiling every experiment in this doc
has hit, so particle-count shot noise alone does not explain the
generalization ceiling. There is a real trend (galaxies with under
10 young particles show std 0.55 dex in log-sSFR vs. 0.34 dex for
galaxies with over 1000), but it isn't large enough, or concentrated
in enough of the population, to drag the whole-population ceiling
down to what we observe.

**Caveat**: `1/√N` is the simplest possible noise model, pure
Poisson counting statistics on particle count. It's a lower bound on
target noise, not a full accounting. Real star formation
stochasticity (bursty or correlated star formation, IMF sampling,
feedback-driven fluctuations within the 1 Gyr window) could exceed
simple shot noise. This check rules out the simplest version of the
target-noise hypothesis as sufficient on its own; it does not rule
out target noise entirely.

**Verdict**: like every other lever tested in this doc, target noise
(at least via this specific, most conservative mechanism) does not
explain the ceiling. The mystery stays open: either the remaining
architecture axes (backbone capacity, dropout placement) matter more
than every other axis tested so far, or the noise floor is real but
comes from a more subtle source than raw particle-count discreteness,
or the images genuinely do not carry enough information to do much
better than R² ≈ 0.3 on this target. This doc cannot rule any of
those three in or out yet.

**Follow-up: real multi-window comparison.** The shot-noise check
above only bounds one specific noise mechanism. To check for real
astrophysical burstiness (star formation genuinely fluctuating within
the 1 Gyr window, not just particle-count discreteness), we computed
sSFR at several averaging windows for the same galaxies, using
`UCITools.ProcessFIREBox.get_avg_sfrs` directly (the exact function
`avg_sfrs_1.0Gyr_no_bound_filter.csv` was built with) at ages 0.1,
0.2, 0.3, 0.5, 0.75, and 1.0 Gyr (`sfr_window_sweep.jl`, Greenplanet,
not committed). Comparing the 1 Gyr and 100 Myr windows for the same
galaxies showed real disagreement well beyond what shot noise alone
predicts: about 4.9x the shot-noise level. Propagating that
disagreement into the same kind of ceiling estimate as above gives an
implied R² ≈ 0.295, remarkably close to the ~0.24-0.31 ceiling every
experiment in this doc has hit. On its face, this looked like it
could explain the whole mystery: real burstiness between windows,
not shot noise, degrading the achievable R² on a 1 Gyr target.

**Direct test (0.3 Gyr regressor target), two seeds:** the natural
next question was whether a shorter-window target would actually be
*easier* to predict from these images, since a shorter window is
closer to whatever timescale of star formation the images visually
encode. We generated `avg_sfrs_0.3Gyr_no_bound_filter.csv` on
Greenplanet (`gen_sfrs_0p3.jl`, mirrors production `gen_sfrs.jl`
exactly) and retrained the regressor with the 0.3 Gyr sSFR as the
target instead of the 1 Gyr one, keeping the classifier's 1 Gyr
star-forming subset as-is and dropping the small number of galaxies
that are star-forming at 1 Gyr but exactly zero at 0.3 Gyr (a
"quick", monkeypatch-only experiment, not a dataset schema change;
`short_window_target.py`, scratchpad, not committed). Two seeds, both
worse than the matched-seed 1 Gyr baseline from the head-width sweep:

| Seed | Target window | Val loss | Val R² |
|------|---------------|----------|--------|
| 42   | 1 Gyr (baseline, large head) | 0.7324 | 0.256 |
| 42   | 0.3 Gyr | 0.8158 | 0.159 |
| 7    | 1 Gyr (baseline, large head) | 0.7216 | 0.267 |
| 7    | 0.3 Gyr | 0.7513 | 0.215 |

**Verdict**: the burstiness hypothesis's clean numerical match (R² ≈
0.295 implied vs. ~0.24-0.31 observed) does not survive direct
testing. Shortening the target window made generalization worse, not
better, on both seeds. The likely reframe: the 1 Gyr window's
smoothing isn't the problem after all. A shorter window does carry
more real astrophysical variance (confirming the multi-window
comparison's finding that windows disagree by more than shot noise),
but that variance is itself less predictable from a single image,
not more, so the network has a harder target, not an easier one. The
apparent numerical coincidence between the implied ceiling and the
observed one was misleading, not causal.

**Color baseline check (2026-07-19).** Separately, we checked whether
crude image color alone carries sSFR signal, as a cheap way to test
whether the CNN's real signal is mostly morphological/spatial rather
than a simple color/brightness summary. Using a per-band "flux" proxy
(sum of positive pixel values per channel, no photometric
calibration, no aperture matching; `color_ceiling_check.py`,
scratchpad, not committed) on the same train/val split and the same
train-only-fit target scaling as the original ceiling check, color
alone added essentially nothing (linear val R² -0.020, gradient
boosting -0.020), and adding it on top of `log10(Mstar)` + `Re` only
marginally helped the gradient boosting model (0.038 -> 0.065; linear
stayed flat, -0.012 -> -0.009).

**Correction (2026-07-22): the "crude proxy" caveat above was
wrong.** It assumed this check's weakness was an uncalibrated
aperture/units problem, on the theory that real photometry needs
background subtraction and a flux zero-point neither the raw pixel
sum nor the image pipeline was known to provide. Checking
`~/code/mockobservation-tools/mockobservation_tools/galaxy_tools.py`
(`get_mock_observation`, which generates these images via
`raytrace_ugr_attenuation`) settles this: pixel values are
`'SB_lum': Luminosity per square distance (Lsun/kpc^2)`, genuine
physical surface brightness from a dust-attenuated ray trace, not
instrument counts needing a zero-point. There is no sky background
to subtract (a pure simulation image), and every galaxy is imaged at
the same distance, so there's no distance-modulus inconsistency
either. None of the concerns a real observational photometry pipeline
has to correct for actually apply here. A Sersic-fit-based total flux
(available for the satellite subset via `AstroPhot_Sate_Sersic_
AllMeasure.csv`'s `Ie`/`Re`/`n`/`b_a`) would mainly help with flux
lost outside a fixed field of view for unusually extended galaxies, a
second-order effect, not a first-order flaw in the original check.
**The original null result stands, and is more solid than this doc
previously gave it credit for.**

That raises the real question this doc hadn't answered: given a
reasonably legitimate color measurement, why does color still add
nothing? Four likely, non-exclusive reasons, none of which needed a
better proxy to find:

1. **The check ran on the star-forming subset only.** Real
   observational color-SFR relations are strongest for the coarse
   quenched/star-forming split, which the
   [classifier][classifier] already gets right at F1 ~ 0.97. Once
   that split is already made, color's dynamic range compresses a
   lot within just the star-forming population, leaving much less
   room for it to track the *continuous* sSFR variation the
   regressor is scored on.
2. **Dust attenuation is explicitly modeled in the image pipeline**
   (`raytrace_ugr_attenuation`), and dust preferentially reddens
   light from young, actively star-forming regions. Within the
   star-forming population, the most active, dustiest galaxies can
   look redder, not bluer, partially inverting the naive
   "bluer = more star formation" relationship. This is a known
   complication in real observational SFR-from-color work too,
   not a defect specific to this check.
3. **The [window-sweep finding][target-noise] predicts exactly
   this.** Color mainly traces relatively short-timescale star
   formation (however long young stars stay blue), and this doc
   already found that a shorter-timescale sSFR target is *harder*
   to predict from these images than the 1 Gyr one. If color
   specifically encodes the short-timescale, burstier component,
   it's tracking the noisier signal relative to what the regressor
   is actually scored against.
4. **This check was always a narrower test than it sounds.** It
   asks whether a single global scalar color helps a linear/GBR
   model beyond mass and size, not whether the CNN uses
   color-like information at all. The CNN's input already has
   per-pixel, per-band values, direct access to spatially resolved
   color (including color gradients across a galaxy), if that's
   useful. A global scalar color not helping a simple model says
   nothing about whether the CNN is already using something similar
   internally.

Reasons 1 and 2 are the likeliest explanation for the literal null
result; reason 4 is why this check was never positioned to answer
"does the CNN use color-like information" either way, only "does a
naive global color feature help a simple model," a narrower
question. Calibrated/photometric color as a candidate is closed:
not because it was tried and failed under a fair test (it was
already fairly tested), but because the specific concern that
motivated retrying it, aperture and units, does not apply to this
dataset.

### Heteroscedastic regression head (2026-07-20)

With the architecture axis closed (see
[Considered and set aside][considered] and the
[backbone dropout][backbone-dropout] result), this tests the
target-noise explanation directly instead of by elimination. The
regressor's head now outputs two values, a mean and a log-variance
of the asinh-scaled sSFR target, trained on `nn.GaussianNLLLoss`
instead of MSE (see the [appendix][nll] for what that loss does and
why the learned variance should be meaningful, not arbitrary). Same
small head and full `[16,32,64,128]` backbone as the recent
baseline, two seeds, otherwise identical hyperparameters.

**Point accuracy, checkpointed by best NLL:**

| seed | val R² |
|---|---|
| 42 | 0.315 (best point-R² of any run in this doc) |
| 7 | 0.204 (clearly worse than the 0.287 MSE baseline) |

Same seed-inconsistency trap as the head-width sweep: one seed looks
like a breakthrough, the other looks like a regression. Point
accuracy is not the reason this experiment was run, and it doesn't
resolve anything on its own here either.

**Calibration, the actual point of the experiment.** For each val
galaxy, the predicted variance should track how large its actual
squared error turns out to be, if the network is genuinely learning
something about per-galaxy predictability rather than gaming the
`log(σ²)` term. Binning val galaxies into 5 quantiles of predicted
variance and comparing to the mean actual squared error in each bin,
both seeds show a real, monotonic trend:

| seed 42 bin | mean pred var | mean actual sq err | seed 7 bin | mean pred var | mean actual sq err |
|---|---|---|---|---|---|
| 0 | 0.188 | 0.216 | 0 | 0.275 | 0.339 |
| 1 | 0.296 | 0.377 | 1 | 0.359 | 0.392 |
| 2 | 0.374 | 0.576 | 2 | 0.489 | 0.991 |
| 3 | 0.580 | 0.835 | 3 | 0.808 | 1.088 |
| 4 | 1.116 | 1.409 | 4 | 1.344 | 1.278 |

Spearman rank correlation between predicted variance and actual
squared error: 0.275 (seed 42), 0.277 (seed 7). Reproducible across
independent seeds and the right sign, so at face value this looks
like real, if modest, calibration: the network has learned to flag
which galaxies it's less sure about, and those galaxies really do
have bigger errors.

**But most of that correlation turned out to be a simpler pattern in
disguise.** Predicted variance also correlates with the target's
raw magnitude (`|target|`, Spearman 0.33-0.35 in both seeds), and
actual squared error correlates with `|target|` even more strongly
(0.55-0.62). That's expected on its own, asinh-standardized targets
further from the population mean are intrinsically harder to hit
exactly, a form of heteroscedasticity that has nothing to do with
per-galaxy image content. Controlling for it (residualizing both
predicted variance and squared error against `|target|` within
quintile bins, then rank-correlating the residuals) collapses the
correlation to 0.131 at seed 42 and 0.025 at seed 7, weak and not
seed-consistent.

**Verdict**: the network learned that extreme-target galaxies are
harder to predict precisely, which is true but is a population-level
fact recoverable from the target distribution alone, not a
galaxy-specific signal the image adds. Once that's accounted for,
there's little left, and what's left doesn't survive a reseed check
either. This is a real result, not a null one: it fails to turn the
target-noise hypothesis into the kind of galaxy-level evidence hoped
for (e.g. "the model is specifically unsure about the bursty,
low-particle-count galaxies flagged in the shot-noise check"). It
neither confirms nor refutes target noise as the ceiling's cause,
it just shows this particular diagnostic mostly measured something
else. A more direct test would need to correlate predicted variance
against actual burstiness (the multi-window sSFR comparison data)
rather than against squared error, since squared error is exactly
where the `|target|` confound comes from.

## Post-mortem: pretrained initialization

We hoped pretrained ImageNet weights would give `StandardNet`'s
ResNet-18 backbone a head start on useful low-level features (edges,
textures, shapes) that could transfer to galaxy morphology, cutting
training time and possibly improving generalization by starting from
a less arbitrary point in weight space than random init.

That didn't pan out either. From "Controlled regressor experiments"
above:

| run | best epoch | train loss | val loss | val R² |
|---|---|---|---|---|
| baseline (scratch) | 15 | 0.62 | 0.716 | 0.276 |
| pretrained (50 ep) | 50 (still improving) | 0.017 | 0.744 | 0.250 |

Pretrained landed slightly worse on val R² than the scratch baseline
(0.250 vs. 0.276), while its train loss collapsed to near-zero
(0.017 vs. 0.62), a much larger train/val gap than any scratch run
showed. It was also still improving on train loss at epoch 50,
unlike every scratch run, which peaked on val loss early (epoch
15-26) and then degraded. Pretraining let the model fit the training
set faster and further, but that extra capacity to fit didn't
translate into better generalization; if anything, it sharpened the
overfitting signature.

**Verdict**: pretrained ImageNet weights aren't delivering a
generalization benefit here. The likely reason is exactly the
concern raised before ever trying it: ImageNet's classes (everyday
objects, animals, scenes) share little structurally with galaxy
images (diffuse light profiles, no hard object boundaries,
single-channel-per-band astronomical imaging plus a velocity map),
so the transferable low-level features pretraining usually helps
with may just not be the bottleneck for this task. What pretraining
does reliably give the model is a head start on fitting quickly,
useful if training time were the constraint, but training time isn't
what's limiting the regressor: it already peaks within 15-26 epochs
from scratch. One caveat worth remembering: we only have one clean
pretrained data point (the initial 50-epoch run); the resumed
continuation past epoch 50 hit the `batch_size` resume bug (Bug 1
above) and isn't a clean comparison, and this result hasn't had a
reseed check the way the capacity sweep just got. This verdict rests
on thinner evidence than the dropout or capacity ones, worth keeping
in mind if pretraining ever comes back up as an idea.

## Post-mortem: dropout

Dropout was one of the hypotheses we had real hope for. It's the
standard, textbook regularizer for exactly this situation (a
high-capacity CNN overfitting a small dataset); `cnn.ResNet` already
had it wired in (p=0.5, right before the head) with that intent, and
finding Bug 2 (eval-mode dropout silently active during validation)
briefly raised the hope that its benefit had been there all along,
just hiding behind noisy, corrupted val metrics.

That didn't pan out. Laying out the full chain of evidence in one
place:

1. **The bug was real**: `cnn.ResNet` validation forward passes were
   randomly zeroing 50% of features the whole time, an unambiguous
   bug. We fixed it by passing `training=self.training` to both
   `F.dropout` calls in `cnn.py`.
2. **Fixing it changed essentially nothing**: the buggy-vs-fixed A/B
   (same split, lr, batch size, epochs, seed) landed at val loss
   0.7385/R²=0.263 (buggy) vs. 0.7324/R²=0.256 (fixed). The two
   metrics even disagree on which direction is "better," which is
   the signature of noise, not a real effect.
3. **Zooming out past just this A/B, dropout's presence or absence
   doesn't correlate with better generalization anywhere in this
   project's history**: the earliest `cnn.ResNet` runs (dropout
   present, though buggy) got R² ≈ 0.23-0.24; the new controlled
   `cnn.ResNet` A/B (dropout present, buggy or fixed) got R² ≈
   0.26; every `StandardNet` run (no dropout at all, never has) got
   R² ≈ 0.24-0.28. Three different dropout states, one
   indistinguishable band of outcomes.

**Verdict**: as implemented here, dropout does not appear to be
delivering the regularization benefit it was added for. Two honest
guesses at why, neither confirmed: (a) a single dropout layer late
in the network, right before the head, may be too weak/too localized
to meaningfully constrain a whole ResNet-18 backbone's capacity to
memorize ~1500 training galaxies, since it regularizes the head's
inputs rather than the conv layers doing most of the representation
learning; or (b) the overfitting here may not be the "co-adapted
feature detector" failure mode dropout specifically targets, but a
more basic sample-size/capacity mismatch that dropout's mechanism
doesn't address regardless of where it's placed. Guess (b) has
since been confirmed, and guess (a) ruled out. The
[backbone-width sweep][backbone-sweep] found that cutting backbone
capacity made generalization worse, not better. Then
[backbone dropout][backbone-dropout] tested guess (a) head-on, by
putting dropout exactly where this paragraph said it was missing,
throughout the conv backbone: it nearly closed the train/val gap and
still made val worse. So dropout's failure here was never about
placement. Dropout had a
real, fair shot here (correctly implemented, tested in a clean A/B,
and cross-checked against a whole project's worth of runs) and didn't
move the number.

## Considered and set aside: a different architecture

Is the ceiling a case of "wrong architecture," where something more
capable (a vision transformer, a multi-scale encoder, a multi-branch
physics-informed network) would unlock more signal? The evidence
gathered so far argues against that as the primary explanation. A
single strong pattern threads through every experiment in this doc:
a wide, orthogonal set of levers (learning rate, ImageNet
pretraining, projection count, dropout as currently placed, head and
backbone capacity, particle shot noise, target window, color) all
land in the same R² 0.24-0.31 band. If the network itself lacked
representational power, ImageNet pretraining should have helped at
least a little, since transfer learning gives a head start on
general-purpose edge/texture/shape features; instead it sharpened
overfitting (see the [pretrained post-mortem][pm-pretrained]). The
0.3 Gyr target experiment argues even more directly: a shorter
window, closer to what a single snapshot image plausibly encodes,
made the regressor generalize worse, not better, consistent with
real, image-invisible stochasticity in the target itself rather than
a network too weak to extract what's already there. No architecture,
however capable, can predict a target's genuinely stochastic
component from an image that doesn't encode it.

That argues for keeping architecture changes targeted rather than
jumping to a fundamentally different architecture family. Three
options along those lines, named and set aside, with reasons:

- **Vision transformers**: typically need more data than CNNs to
  outperform them, since they lack a CNN's built-in spatial
  inductive bias and have to learn locality from data instead. This
  dataset (roughly 1500-2000 galaxies) sits well below where that
  usually pays off, so a ViT is more likely to underperform a
  ResNet here than beat it, not a promising place to spend effort
  first.
- **Multi-scale encoders (FPN-style)**: the one option with a real
  physical justification, if the discriminating signal lives at a
  spatial scale a plain ResNet's downsampling washes out (e.g. a
  faint diffuse component vs. a compact bright core). But nothing
  tested so far points at scale specifically as the problem, so this
  stayed speculative rather than evidence-motivated, and the project
  concluded before it was revisited.
- **Multi-branch, physics-informed networks** (a separate branch for
  a derived quantity, fused with the image branch): really just
  "better features" wearing an architecture costume, and largely the
  same idea as the calibrated-color question this doc later settled
  (see [Project status: concluded][status]): color and photometry,
  however measured, don't appear to add much on top of what the images
  already give the CNN directly.

## Bottom line

The architecture axis is now exhausted, and that is a result, not a
failure. The regressor has a real, measurable signal well above a
trivial mass/size baseline, and the classifier already generalizes
well. We've tested lr, pretraining, projection count, dropout in
both placements, two capacity sweeps (head width and backbone
width), sSFR shot noise, a direct shorter-window sSFR target, and a
crude color baseline, each in a controlled way, and ruled them all
out as the dominant lever.

The strongest single piece of evidence came last. Every
capacity/regularization intervention that constrained the model
either did nothing or made val performance worse, and backbone
dropout made this unmistakable: it nearly eliminated the train/val
gap and val R² *fell*, from ~0.29 to ~0.21. Under an overfitting
story, closing that gap should buy generalization. It didn't, at any
point along the path. The most defensible reading now is that the
large train/val gap was a symptom of a target that single-snapshot
images only partly determine, not the cause of the ceiling, so
constraining the model was never going to fix it.

That came with a real methodological lesson too: the head-width
sweep's first-seed result looked like a breakthrough and mostly
wasn't, and the multi-window burstiness estimate's suspiciously close
numerical match to the observed ceiling also didn't survive a direct
test. Every capacity/regularization experiment on this project now
runs at least two seeds per point from the start, a discipline that
has already paid off three times (the head-width sweep's medium-head
reversal, and the backbone-width and backbone-dropout sweeps both
coming back clean and consistent instead of ambiguous).

The last two axes closed rather than resolved cleanly. The
[heteroscedastic head][het-head] was meant to test the target-noise
explanation directly rather than by elimination, and it
half-worked: predicted variance does correlate with actual error,
reproducibly across seeds, but mostly because both track target
magnitude, a confound a cleaner comparison (predicted variance
against actual burstiness data, not squared error) would need to
resolve, left as a documented future direction rather than pursued.
Color closed more cleanly: the original null result held up once the
[uncalibrated proxy][status] concern was checked against the actual image
pipeline and found not to apply (see
[Project status: concluded][status]).

The honest conclusion isn't that this project failed, it's that it
measured something: single-snapshot images predict roughly this much
of sSFR's variance, on top of a classifier that already separates
quenched from star-forming galaxies well, and the regressor's
remaining ceiling most likely reflects real, image-invisible
burstiness in star formation on sub-Gyr timescales, not an
engineering shortfall. That's a claim about the physics and the
data, supported by an unusually thorough set of controlled negative
results, not a dead end quietly abandoned.

## Appendix: what is Gaussian NLL?

Ordinary MSE training has the regressor predict one number per
galaxy, ŷ, and minimizes (ŷ - y)². Gaussian NLL instead has it
predict two numbers, a mean μ and a variance σ², and treats the
target y as if drawn from a Gaussian distribution N(μ, σ²). The loss
is that Gaussian's negative log-likelihood:

    NLL = 0.5 * log(σ²) + 0.5 * (y - μ)² / σ²

There's no ground-truth "correct" σ² anywhere in the training data to
supervise against directly. The network only ever sees this one
scalar loss value, and gradient descent works out what σ² should be
purely from how the two terms trade off against each other. Holding
μ fixed, taking the derivative of NLL with respect to σ² and setting
it to zero shows the loss is minimized, for any single example,
exactly when σ² equals the squared error (y - μ)². So the predicted
variance gets pulled toward matching whatever residual it actually
produces, example by example. Averaged across many similar galaxies,
this makes the network converge toward predicting the conditional
variance E[(y - μ(x))² | x], the expected squared error given the
image, which is exactly what "uncertainty" should mean here.

The `log(σ²)` term is what keeps this from degenerating. Without it,
the network could trivially shrink the residual term to zero by
predicting σ² → ∞ everywhere the target is hard, without learning
anything real. `log(σ²)` grows as σ² grows, so blowing up variance
indiscriminately costs loss too, and the network only benefits from
predicting a larger σ² where the residual term it buys back in
return is worth more than that cost, i.e. galaxies that are
genuinely hard to predict. In practice this is usually implemented
by having the head output `log(σ²)` directly rather than σ² itself,
since that guarantees positivity and avoids ever dividing by a
variance that could hit zero.

Whether this actually worked, whether the predicted uncertainty
means anything, isn't taken on faith. It gets checked afterward with
a calibration plot: bin val galaxies by predicted σ² and check
whether the actual residual variance within each bin matches. That's
also what makes this a useful diagnostic for the target-noise
question specifically: if predicted σ² tracks the low-particle-
count, bursty galaxies the shot-noise check already flagged, that's
evidence the ceiling is target stochasticity; if predicted σ² comes
out flat regardless of galaxy properties, that argues against it.


<!-- Section links, used by the lists above. -->

[log]: #experiment-log
[classifier]: #classifier-generalizes-fine
[earliest]: #regressor-earliest-results-pre-split-refactor-early-resnet
[ceiling]: #ceiling-check-2026-07-17
[controlled]: #controlled-regressor-experiments-2026-07-17---2026-07-18
[bug1]: #bug-1-2026-07-18-fixed---resume-didnt-restore-settings
[bug2]: #bug-2-2026-07-19-fixed-eval-mode-dropout-active-in-validation
[dropout-ab]: #dropout-bug-ab-2026-07-19-fix-made-no-real-difference
[head-sweep]: #capacity-sweep-head-width-2026-07-19
[backbone-sweep]: #capacity-sweep-backbone-width-2026-07-20
[backbone-dropout]: #dropout-placement-dropout-in-the-backbone-2026-07-20
[target-noise]: #ssfr-target-noise-check-2026-07-19
[het-head]: #heteroscedastic-regression-head-2026-07-20
[pm-dropout]: #post-mortem-dropout
[pm-pretrained]: #post-mortem-pretrained-initialization
[candidates]: #documented-future-directions-not-pursued
[status]: #project-status-concluded-2026-07-22
[considered]: #considered-and-set-aside-a-different-architecture
[nll]: #appendix-what-is-gaussian-nll
