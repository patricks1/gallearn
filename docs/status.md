# Project status: sSFR regression generalization

Living document tracking what we know about why the sSFR regressor
isn't generalizing well, what we've tried, and what's next. This is
also a historical log: append new findings, don't delete or silently
rewrite old ones. If something turns out to be wrong or confounded,
say so in place and point forward, rather than removing it.

Last updated: 2026-07-19.

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

We've tested lr, pretraining, and projection count each in a
controlled way (see log below), and every one of them lands in the
same narrow R² band, with every non-pretrained run peaking very
early (epoch 15-26 of 50) then degrading. That consistency itself is
the finding: none of those three is the bottleneck. The working
hypothesis is now model capacity relative to the ~1500-galaxy
dataset; see "Next candidates" at the bottom.

We found and fixed two real infrastructure bugs along the way
(resume not restoring most run settings; eval-mode dropout silently
active during validation); see the log for what they affected.

## Experiment log

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

### Regressor: earliest results (pre-split-refactor and early
`--model resnet` runs)

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

### Bug 1 (found 2026-07-18, fixed): `--resume` silently didn't
restore most run settings

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

### Bug 2 (found 2026-07-19, fixed): eval-mode dropout was silently
active during validation

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
doesn't address regardless of where it's placed. Either way, this is
why the next candidate is capacity reduction throughout the network
(`cnn.ResNet`'s `n_blocks_list`/`out_channels_list`), not more
dropout tuning. Dropout had a real, fair shot here (correctly
implemented, tested in a clean A/B, and cross-checked against a
whole project's worth of runs) and didn't move the number.

## Next candidates, not yet started

- **Capacity-reduction sweep on `cnn.ResNet`**: the model/data
  capacity mismatch (~11M-parameter ResNet-18 for `StandardNet`, or
  even `cnn.ResNet`'s smaller custom net, fit to ~1500 distinct
  galaxies) is the most likely remaining explanation for the
  early-peak-then-degrade pattern seen everywhere above.
  `cnn.ResNet`'s `n_blocks_list`/`out_channels_list` are the tool
  for this (`StandardNet` has no way to customize block count or
  channel width, since it's locked to whatever `torchvision.models.
  resnet18()` gives). Note pretraining and capacity reduction don't
  compose: shrinking a stock ResNet-18 below its stock depth/width
  makes the pretrained ImageNet weights no longer fit, so this stays
  a `cnn.ResNet`-only, scratch-only experiment. Run this before the
  dropout-placement idea below, since it's the cleaner test of the
  underlying capacity hypothesis, and it also tells us whether
  backbone-level dropout is even worth trying next.
- **Dropout placement**: the post-mortem above found that fixing the
  eval-mode dropout bug made no real difference, but the working
  hypothesis (see "Verdict" in the post-mortem) is that the single
  `p=0.5` dropout layer in `cnn.ResNet`, applied only right before
  the head, may be too weak or too localized to constrain the conv
  backbone where most of the capacity to memorize training galaxies
  actually lives. If the capacity-reduction sweep above doesn't fully
  resolve generalization on its own, try moving dropout into the conv
  backbone itself (e.g. spatial/2D dropout between residual blocks,
  not just before the head) as a different mechanism for constraining
  a backbone that shrinking alone didn't fix.
- Multiple val splits (`scripts/split.py split --split-name ...`) to
  get a variance estimate on val R², since a single 85/15 galaxy
  split with ~1500 galaxies is noisy on its own, and the differences
  between rows in the controlled-experiments table above (0.71-0.76)
  are narrow enough that some could just be single-seed noise.

## Bottom line

Not futile. The regressor has a real, measurable signal well above a
trivial mass/size baseline, and the classifier already generalizes
well. We've tested lr, pretraining, and projection count each in a
controlled way and ruled them out as the dominant lever: every one
of them lands in the same R² ≈ 0.24-0.28 band. That's useful negative
information. The next real candidate is model capacity relative to
the ~1500-galaxy dataset size, not further optimizer tuning.
