# Project status: overview

Living summary of where this project stands. See [sSFR experiments](status_sfr.md) for the full sSFR regression experiment log and [fgas and fdm experiments](status_fgas_fdm.md) for the fgas and fdm experiment log. This file is the entry point: current results, the key cross-target finding, and links forward.

## Current flagship models

Best val R² and intuitive-space MAE at best checkpoint, two seeds, from the most recent clean experiment per target. MAE units differ by target (dex for avg_sfr, raw gas/dark-matter fraction for fgas/fdm), so MAE does not compare across targets. The galfiltered and unfiltered rows are different val populations and do not compare either; note that the galfiltered models reach lower R² yet smaller MAE, the variance artifact described in the [fgas and fdm log](status_fgas_fdm.md).

**Galfiltered.** fdm R² is raw-space (scaled logit R² is not comparable).

| target | architecture | head | R² (s42) | R² (s7) | MAE (s42) | MAE (s7) | source |
|--------|--------------|------|----------|---------|-----------|-----------|--------|
| avg_sfr | ResNet | small (128..8) | 0.361 | 0.392 | 0.192 dex | 0.189 dex | [galfiltered avg_sfr][galfiltered-avgsfr] |
| fgas | ResNet | minimal (64/32) | 0.520 | 0.509 | 0.0175 | 0.0176 | [galfiltered fgas/fdm][galfiltered-runs] |
| fdm | ResNet | minimal (64/32), logit | 0.233 | 0.147 | 0.0358 | 0.0355 | [galfiltered fgas/fdm][galfiltered-runs] |

**Unfiltered.**

| target | architecture | head | R² (s42) | R² (s7) | MAE (s42) | MAE (s7) | source |
|--------|--------------|------|----------|---------|-----------|-----------|--------|
| avg_sfr | ResNet | minimal (64/32, ~321K params) | 0.302 | 0.284 | 0.235 dex | 0.240 dex | [capacity sweep][head-sweep-sfr] |
| fgas | ResNet | minimal (64/32, ~10K params) | 0.6228 | 0.6154 | 0.0367 | 0.0359 | [head-size sweep][head-sweep-new] |
| fdm | ResNet | minimal (64/32, ~10K params) | 0.7631 | 0.7694 | 0.0641 | 0.0658 | [head-size sweep][head-sweep-new] (pre-logit) |

## Project status: the vmap carries the signal (2026-09-08)

The first real experiments on the new targets landed a result that outranks everything else measured on them. Gas fraction and dark-matter fraction both look easy compared to sSFR, at val R2 ~0.62 and ~0.77 against an sSFR ceiling of ~0.31. Both numbers come entirely from one input channel. Zero the velocity map and performance on both targets falls to zero, reproducibly, on both seeds:

| target | seed | with vmap | vmap zeroed | change |
|---|---|---|---|---|
| fgas | 42 | 0.6150 | -0.0000 | -0.6151 |
| fgas | 7 | 0.6110 | 0.0062 | -0.6049 |
| fdm | 42 | 0.7585 | 0.0008 | -0.7577 |
| fdm | 7 | 0.7728 | 0.0048 | -0.7680 |

Not degraded, gone. The three ugr bands contribute nothing this pipeline can measure on either target. The ablated models still train (their train loss falls), so nothing crashes; they simply learn nothing that transfers to val galaxies. For fdm the ablated network lands below the mass-and-size scalar baseline of 0.354, even though it receives Re directly and could in principle read stellar mass off the ugr brightness.

What the vmap actually contains explains why this matters. `UCITools`' `vel_map.calc_vmap` builds a gas column-density map, keeps only bins above `min_cden`, and writes NaN elsewhere:

    mask = cden_map >= min_cden
    vmap = np.where(mask, v_y_colormap, np.nan)

`LazyGalaxyDataset` then turns those NaNs into zeros. So the channel bundles two different things. Its non-zero footprint is a map of where this galaxy holds gas above a threshold, and the values inside that footprint are gas kinematics tracing the gravitational potential.

Those two components carry very different weight for the two targets. Predicting gas fraction from the extent of a gas map approaches circularity, since the input derives from the same physical quantity as the target. Predicting dark-matter fraction from a velocity field is instead what rotation curves have done for decades, so it may be real kinematic mass modeling rather than leakage. The current ablation cannot separate these, because zeroing the channel destroys the footprint and the kinematics together.

Either way, one claim is already ruled out. These are not results about what galaxy images show. Photometry contributes nothing here. Any framing of this work should say that gas velocity maps predict these quantities, not that images do.

**The experiment that resolves it** splits the channel's two components, at head-small, two seeds each:

- **Mask only**: replace the velocity values with a binary gas-presence mask. Keeps the footprint, destroys kinematics.
- **Kinematics only**: keep velocity values but break the footprint's correspondence to gas extent, by shuffling values spatially within the mask or randomizing the mask's extent while preserving the velocity distribution.

If fdm survives kinematics-only and dies on mask-only, the dark-matter result is physics. If both targets follow the mask, both are measuring gas extent and the headline numbers do not mean what they appear to.


<!-- Section links -->

[backbone-dropout]: status_sfr.md#dropout-placement-dropout-in-the-backbone-2026-07-20
[head-sweep-new]: status_fgas_fdm.md#head-size-sweep-on-the-new-targets-2026-09-08
[head-sweep-sfr]: status_sfr.md#capacity-sweep-head-width-2026-07-19
[galfiltered-avgsfr]: status_sfr.md#galfiltered-training-avg_sfr-2026-10
[galfiltered-runs]: status_fgas_fdm.md#galfiltered-training-first-runs-2026-10
