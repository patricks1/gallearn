"""
One-time migration: inject target_stats['method'] into checkpoints
that predate self-describing scaling.

Each entry in MIGRATIONS maps a checkpoint directory to the method
string that was active when those checkpoints were trained. The
script patches every checkpoint_epoch*.pt file in each directory;
best.pt is a symlink and is skipped (it resolves to one of those
files).

Run once after updating gallearn/target_specs.py to the dispatch-
based scaling. Safe to run again: if a checkpoint already has the
key, the script reports it and leaves it unchanged.
"""
import pathlib

import torch

import gallearn.config


def _models_dir():
    return pathlib.Path(
        gallearn.config.config['gallearn_paths']['models_dir']
    )


MIGRATIONS = {
    #'fdm_head-small_seed7': 'plain_std',
    #'fdm_head-small_galfiltered_logit_seed7': 'logit_std',
    #'fgas_head-minimal_galfiltered_seed7': 'plain_std',
    #'fgas_head-minimal_galfiltered_svlogit_seed7': 'sv_logit_std',
    #'fgas_head-minimal_galfiltered_svlogit_lr3e4_seed7': 'sv_logit_std',
    #'fdm_head-small_galfiltered_seed7': 'plain_std',
    #'heteroscedastic_head_seed42': 'asinh_std',
    #'fgas_head-minimal_seed42': 'plain_std',
    #'fdm_head-minimal_seed7': 'plain_std',
    #'fgas_head-minimal_seed7': 'plain_std',
    #'fdm_head-minimal_seed42': 'plain_std',
}


def patch_directory(directory, method):
    d = _models_dir() / directory
    if not d.is_dir():
        print('  SKIP (directory not found): {}'.format(d))
        return
    files = sorted(d.glob('checkpoint_epoch*.pt'))
    if not files:
        print('  SKIP (no epoch checkpoints): {}'.format(d))
        return
    for path in files:
        checkpoint = torch.load(path, weights_only=False)
        target_stats = checkpoint.get('target_stats')
        if target_stats is None:
            print('  SKIP (no target_stats, classifier?): {}'.format(
                path.name
            ))
            continue
        existing = target_stats.get('method')
        if existing is not None:
            print('  already set to {!r}: {}'.format(
                existing, path.name
            ))
            continue
        target_stats['method'] = method
        torch.save(checkpoint, path)
        print('  patched -> {!r}: {}'.format(method, path.name))


if __name__ == '__main__':
    for directory, method in MIGRATIONS.items():
        print(directory)
        patch_directory(directory, method)
