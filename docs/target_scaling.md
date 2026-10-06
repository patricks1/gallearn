# Target variable scaling

This document covers how the pipeline scales each target before training
and why those choices diverge from the visualization scale
(`plot_scale` in `target_specs.py`).

## The two scaling systems are independent

`TargetSpec` carries two completely separate scaling concerns that happen
to sound similar.

**Training scale** (`fit` / `scale` / `unscale`) transforms raw target
values into a form the network trains on. The goals are a well-behaved
gradient signal and roughly unit-variance targets, so the optimizer
doesn't have to fight wildly different loss magnitudes across batches.
Only `train.py` and `evaluate.py` (for unscaling predictions) use this.

**Plot scale** (`plot_scale`) controls how scatter plots and distribution
panels display the unscaled values. It is a visualization choice that
applies after training finishes and has no effect on what the network
sees.

Choosing them together would be a mistake. A logit transform might show
a fraction's distribution well (spreading out both tails) without being
a good training transform. A log transform might be necessary for
training (because raw values span orders of magnitude) without being
the right axis for a scatter plot. They answer different questions.

## Per-target choices

### sSFR

Raw sSFR on the star-forming subset spans roughly ten orders of
magnitude. Plain standardization on raw values would assign variance
almost entirely to the extreme high end of the distribution, and the
network would spend most of its capacity predicting which rare galaxies
are very highly star-forming rather than learning the structure of the
middle of the population.

`SsfrTarget` applies `asinh(stretch * x)` before standardizing, with
`stretch = 1e11`. Asinh compresses large values logarithmically while
staying defined and smooth all the way down to zero (unlike log), so
the star-forming tail shrinks to a manageable range without requiring
any special handling of edge cases. Standardization on top of that
centers and scales the compressed values to roughly unit variance for
the optimizer. We set `plot_scale = 'log'` independently. Once the
structural zeros are excluded (the regressor only sees star-forming
galaxies), a log axis reads more naturally than a linear one.

### Gas fraction

Gas fraction spans a narrow range, with the bulk of the population
below 0.2 and a long thin tail out toward one. Plain standardization
suffices; the range is not wide enough to need a stretch.

The exact zeros rule out log and logit transforms for training. About
1.5% of galaxies have exactly zero gas fraction, a real physical value
(fully stripped halos), and log and logit are both undefined at zero.
Using either as a training transform would require dropping those
galaxies or clipping them to some near-zero substitute, both of which
introduce bias. Standardization handles them cleanly. We set
`plot_scale = 'log'` for visualization, where the distribution panel
breaks exact zeros out as a separate bar rather than feeding them to
the transform.

### Dark-matter fraction

The galfiltered fdm dataset (12,339 galaxies) has no near-zero
galaxies at all. The galaxy-only filter removed the entire low-fdm
cluster. The distribution now runs from a minimum of about 0.33 to a
maximum of 0.999, with a median of 0.90 and 99.7% of galaxies above
0.5. It is unimodal, concentrated near 1, with a tail that builds
sharply from about 0.83 upward.

Training uses logit then standardize. The diagnostic that motivated
the change was predictions clustering between 0.8 and 0.96, a hard
visible boundary in the scatter plot. With plain standardization, MSE
gradient for the dense near-1 tail is negligible: fdm=0.96 vs
fdm=0.99 differ by only 0.03 in linear space, so the model regresses
to the mean of the bulk distribution. Logit spreads that tail
dramatically (logit(0.96) ≈ 3.18, logit(0.99) ≈ 4.60,
logit(0.999) ≈ 6.91, roughly 48x more spread than linear over that
interval), giving the optimizer a well-resolved gradient signal
throughout the distribution. Since no galaxy sits at exactly 0 or 1,
logit is well-defined on every training sample.

The operational complication is that training metrics (RMSE, MAE, R²)
are in log-odds units during a run. To compare against a
plain-standardization baseline, unscale predictions back to fdm units
before computing metrics.

`plot_scale = 'logit'` spreads the near-1 tail for visualization and
keeps scatter plots legible.

## Summary

| target | training transform | notes |
|---|---|---|
| sSFR | asinh + standardize | asinh handles the multi-order range and stays defined at zero |
| fgas | standardize | exact zeros are undefined under log/logit |
| fdm | logit + standardize | spreads the near-1 concentration; metrics are in log-odds units |
