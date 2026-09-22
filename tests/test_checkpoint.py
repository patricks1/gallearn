'''
Unit tests for train.update_best_symlink.
'''
import os

from gallearn import train


def test_symlink_points_at_the_given_checkpoint(tmp_path):
    '''Verify that best.pt resolves to the checkpoint passed in,
    by relative filename rather than an absolute path, so the
    run directory stays relocatable.'''
    ckpt_path = tmp_path / 'checkpoint_epoch005.pt'
    ckpt_path.write_bytes(b'stub')

    train.update_best_symlink(str(tmp_path), str(ckpt_path))

    link_path = tmp_path / 'best.pt'
    assert os.readlink(link_path) == 'checkpoint_epoch005.pt'
    assert link_path.resolve() == ckpt_path


def test_symlink_updates_to_a_later_checkpoint(tmp_path):
    '''Verify that a second call repoints best.pt rather than
    failing because the link already exists, matching how
    train.main() calls this after every new-best epoch.'''
    first = tmp_path / 'checkpoint_epoch005.pt'
    second = tmp_path / 'checkpoint_epoch012.pt'
    first.write_bytes(b'stub')
    second.write_bytes(b'stub')

    train.update_best_symlink(str(tmp_path), str(first))
    train.update_best_symlink(str(tmp_path), str(second))

    link_path = tmp_path / 'best.pt'
    assert os.readlink(link_path) == 'checkpoint_epoch012.pt'
