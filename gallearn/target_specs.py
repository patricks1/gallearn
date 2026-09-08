"""
Per-target behavior for the training pipeline.

A dataset holds exactly one target, and `src/Dataset.jl` records
which one in the HDF5 root attribute `tgt_type`. Everything that
varies from one target to the next lives in a TargetSpec here, so
the shared pipeline in train.py stays free of per-target branches
and adding a target means adding a class plus a REGISTRY entry.

The scaling statistics a spec produces are opaque: only the spec
that created them interprets them. Callers store them in a
checkpoint and hand them back to `scale` and `unscale` without
reading their keys, which is what keeps target-specific detail from
leaking back into the pipeline.
"""
import torch

from . import preprocessing


class TargetSpec:
    """
    Base class describing how to train on one target.

    Attributes
    ----------
    name : str
        Human-readable target name, used in messages.
    column : str
        Column `src/Dataset.jl` read this target from. Provenance
        only; the HDF5 array is always `ys_sorted`.
    supports_classifier : bool
        Whether the quenched/star-forming hurdle classifier applies
        to this target.
    axis_label : str
        How to name this target on a plot axis.
    unit : str
        Unit to show beside axis_label, matplotlib mathtext
        allowed. Empty for a dimensionless target.
    log_scale : bool
        Whether raw values of this target want log axes. A target
        that can legitimately be zero has to say False, since a log
        axis cannot show those points.
    """

    name = None
    column = None
    supports_classifier = False
    axis_label = None
    unit = ''
    log_scale = False

    def select_valid(self, vals):
        """
        Return the row indices to train on, given the raw target
        values (shape (N, 1)).
        """
        raise NotImplementedError

    def population_summary(self, vals):
        """
        Return a one-line description of the training population,
        which prepare_targets prints.
        """
        raise NotImplementedError

    def fit(self, vals):
        """
        Fit scaling statistics on `vals` and return them as an
        opaque dict. Callers pass this to `scale` and `unscale`
        without interpreting it.
        """
        raise NotImplementedError

    def scale(self, vals, stats):
        """Apply the scaling described by `stats` to `vals`."""
        raise NotImplementedError

    def unscale(self, scaled, stats):
        """Invert `scale`, returning values in raw target units."""
        raise NotImplementedError

    def classifier_targets(self, vals):
        """
        Return binary classifier targets, shape (N, 1). Only defined
        when supports_classifier is True.
        """
        raise NotImplementedError(
            '{0} does not support the classifier task.'.format(
                self.name
            )
        )


class SsfrTarget(TargetSpec):
    """
    Specific star formation rate.

    sSFR has a structural zero: a real fraction of galaxies are
    quenched rather than merely low. That is what the hurdle model
    exists for, so the classifier applies here and the regressor
    trains on the star-forming subset alone. Raw sSFR spans many
    orders of magnitude, so scaling runs through asinh before
    standardizing.
    """

    name = 'ssfr'
    column = 'ssfr'
    supports_classifier = True
    axis_label = 'sSFR'
    unit = 'yr$^{-1}$'
    # The regressor only ever sees star-forming galaxies, so every
    # ground-truth value it handles is strictly positive.
    log_scale = True
    stretch = 1.e11

    def _star_forming(self, vals):
        return (vals > 0).squeeze()

    def select_valid(self, vals):
        return torch.where(self._star_forming(vals))[0]

    def population_summary(self, vals):
        n_star_forming = self._star_forming(vals).sum().item()
        n_quenched = len(vals) - n_star_forming
        return (
            'Class balance: {0:.0f} quenched images,'
            ' {1:.0f} star-forming images'.format(
                n_quenched, n_star_forming
            )
        )

    def fit(self, vals):
        _, means, stds = preprocessing.std_asinh(
            vals,
            self.stretch,
            return_distrib=True,
        )
        return {'means': means, 'stds': stds, 'stretch': self.stretch}

    def scale(self, vals, stats):
        return preprocessing.std_asinh(
            vals,
            stats['stretch'],
            means=stats['means'],
            stds=stats['stds'],
        )

    def unscale(self, scaled, stats):
        # std_asinh computes scaled = (asinh(stretch * x) - means)
        # / stds, so the inverse is
        # x = sinh(scaled * stds + means) / stretch.
        return torch.sinh(
            scaled * stats['stds'] + stats['means']
        ) / stats['stretch']

    def classifier_targets(self, vals):
        return self._star_forming(vals).float().unsqueeze(-1)


class FgasTarget(TargetSpec):
    """
    Gas fraction: gas mass over total halo mass, dark matter
    included.

    About 1.5% of galaxies sit at exactly zero, fully stripped of
    gas (their Mgas is exactly zero too, so this is a real physical
    value and not a missing-data sentinel). That is far too small a
    population to justify a hurdle stage the way sSFR's much larger
    quenched fraction does, and those galaxies carry real signal, so
    the regressor trains on all of them. Gas fraction also spans a
    narrow range, so plain standardization replaces sSFR's asinh
    step. Standardizing keeps the exact zeros well behaved, which a
    log or logit transform would not.
    """

    name = 'fgas'
    column = 'fgas'
    supports_classifier = False
    axis_label = 'gas fraction'
    unit = ''
    # Gas-free galaxies sit at exactly zero, which a log axis would
    # drop silently.
    log_scale = False

    def select_valid(self, vals):
        return torch.arange(len(vals))

    def population_summary(self, vals):
        n_zero = (vals == 0).sum().item()
        return (
            'Gas fraction: {0:.0f} galaxy images,'
            ' {1:.0f} of them gas-free'.format(len(vals), n_zero)
        )

    def fit(self, vals):
        _, means, stds = preprocessing.std_scale(
            vals,
            return_distrib=True,
        )
        return {'means': means, 'stds': stds}

    def scale(self, vals, stats):
        return preprocessing.std_scale(
            vals,
            means=stats['means'],
            stds=stats['stds'],
        )

    def unscale(self, scaled, stats):
        # std_scale computes scaled = (x - means) / stds.
        return scaled * stats['stds'] + stats['means']


class FdmTarget(TargetSpec):
    """
    Dark-matter fraction: dark-matter mass over total halo mass.

    Unlike gas fraction, no galaxy sits at exactly zero or exactly
    one, so there is no edge case to carve out. The regressor trains
    on every galaxy, scaled the same way as gas fraction: plain
    standardization, since the range is narrow enough that no
    stretch is needed and a log axis would be pointless without any
    zeros to worry about.
    """

    name = 'fdm'
    column = 'fdm'
    supports_classifier = False
    axis_label = 'dark-matter fraction'
    unit = ''
    log_scale = False

    def select_valid(self, vals):
        return torch.arange(len(vals))

    def population_summary(self, vals):
        return 'Dark-matter fraction: {0:.0f} galaxy images'.format(
            len(vals)
        )

    def fit(self, vals):
        _, means, stds = preprocessing.std_scale(
            vals,
            return_distrib=True,
        )
        return {'means': means, 'stds': stds}

    def scale(self, vals, stats):
        return preprocessing.std_scale(
            vals,
            means=stats['means'],
            stds=stats['stds'],
        )

    def unscale(self, scaled, stats):
        # std_scale computes scaled = (x - means) / stds.
        return scaled * stats['stds'] + stats['means']


# Keyed by the `tgt_type` src/Dataset.jl writes into the HDF5 root
# attributes, which is also what --target accepts. "sfr" and
# "avg_sfr" are both sSFR, differing only in the averaging window
# the Julia side used, which does not change how training treats
# them.
REGISTRY = {
    'sfr': SsfrTarget(),
    'avg_sfr': SsfrTarget(),
    'fgas': FgasTarget(),
    'fdm': FdmTarget(),
}


def get(tgt_type):
    """
    Look up the TargetSpec for a `tgt_type`.

    Parameters
    ----------
    tgt_type : str
        A key of REGISTRY: 'sfr', 'avg_sfr', 'fgas', or 'fdm'.

    Returns
    -------
    TargetSpec

    Raises
    ------
    ValueError
        If no spec covers `tgt_type`.
    """
    if tgt_type not in REGISTRY:
        raise ValueError(
            "No target spec for tgt_type '{0}'. Known targets:"
            ' {1}.'.format(
                tgt_type, ', '.join(sorted(REGISTRY))
            )
        )
    return REGISTRY[tgt_type]
