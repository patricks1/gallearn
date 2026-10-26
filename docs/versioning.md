# Versioning and releases

`gallearn` uses a declarative `pyproject.toml` and
[setuptools_scm](https://setuptools-scm.readthedocs.io/) for versioning.
You never edit a version string by hand anywhere in the source tree.

## Where the version comes from

setuptools_scm derives the version from the repository's git tags at
build time:

- On a commit that is exactly a tag `vX.Y.Z`, the version is `X.Y.Z`.
- On a commit after the latest tag, it is a development version such as
  `X.Y.(Z+1).devN+g<hash>`, where `N` counts the commits since the tag.

setuptools records that version in the installed package's metadata, and
`gallearn.__version__` reads it back at runtime through
`importlib.metadata`, so the attribute always matches the version you
installed. An editable install (`pip install -e .`) freezes the version
at install time, so re-install if you want `__version__` to reflect new
commits.

## How to cut a release

1. Commit the changes you want to release.
2. Tag that commit with a `v`-prefixed PEP 440 version, then push the
   tag:

       git tag v0.1.0
       git push origin v0.1.0

   Pre-releases take an `a`/`b`/`rc` suffix, for example `v0.1.0b1`.
3. That is the whole release. Anyone installing from that tag gets the
   clean version; anyone on a later commit gets a development version
   that sorts above it.

Do not move a tag once you have pushed it and something may have
installed from it. To correct a release, cut a new tag.
