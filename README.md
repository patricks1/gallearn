# GalLearn

Predict star formation in FIREbox galaxies from mock images with a
convolutional neural network.

GalLearn turns simulated galaxy images (plus stellar velocity maps and
Sersic radii) into an HDF5 training set and trains a CNN to (1) classify
galaxies as star-forming or quenched and (2) regress the specific star
formation rate (sSFR) of the star-forming ones.

The repo spans two languages:

- **Julia** builds the training dataset (`src/`, `scripts/*.jl`).
- **Python** defines and trains the network (`gallearn/`).

## Pipeline

1. **Generate inputs** (on Greenplanet).
   - Mock band images per galaxy and projection.
   - Sersic fits giving the effective radius `Re` per projection
     (`scripts/gen_octant_shapes.py`; host/satellite shape CSVs).
   - Average star formation rates (`scripts/gen_sfrs.jl`).
2. **Build the dataset** (Julia). `scripts/transform_images.jl` reads the
   images, velocity maps, and shape/SFR tables and writes one HDF5 file:
   the image tensor `X` (image bands plus a velocity-map channel), the
   `ssfr` targets, and `Re`.
3. **Split the dataset** (Python). Lock the dataset by content hash,
   create or top up the locked test set, and write a train/val split
   (see "Splitting the dataset" below).
4. **Train** (Python). Load the HDF5 dataset and the train/val split,
   and train the CNN.

## Configuration

Runtime paths (image directories, vmap directory, shape CSVs, dataset
name, ...) live in an INI file in your home directory, outside the repo.
Julia and Python locate it differently.

**Julia** (`src/GalLearnConfig.jl`) walks up from the active project and
uses the `name` field of the nearest ancestor `Project.toml`, reading
`~/config_<name>.ini`. The root project is `name = "GalLearn"`, and
sub-projects like `scripts/` deliberately omit `name`, so they inherit
it: `~/config_GalLearn.ini`. There is no environment-variable override.

**Python** (`gallearn/config.py`) checks the `GALLEARN_CONFIG`
environment variable first; if set, it reads that path (CI uses this, via
`conftest.py`, which points it at `.github/config_ci.ini`). Otherwise it
reads `~/config_<conda-env>.ini` (just `~/config.ini` in the base env),
creating it with Greenplanet defaults if it does not exist.

These two schemes produce different filenames, so you have to reconcile
them by hand. In the `gallearn` conda env, Python looks for
`~/config_gallearn.ini` while Julia looks for `~/config_GalLearn.ini`.
Create the config under one name and symlink the other to it yourself so
both read the same file:

    # create and edit ~/config_gallearn.ini, then:
    ln -s ~/config_gallearn.ini ~/config_GalLearn.ini

If you also use UCITools, do the same for it:

    ln -s ~/config_uci_tools.ini ~/config_UCITools.ini

The INI supports `${section:key}` interpolation, e.g.
`dataset = ${paths:dataset}`.

Note: because Python keys the filename off the active conda environment,
keep that environment name consistent across runs, or you will end up
reading (or auto-creating) a different config file.

## Building the dataset

The build runs as a Slurm job. `scripts/transform_images.jl` spawns
Distributed workers to read the HDF5 image files in parallel, assembles
the image tensor, attaches each galaxy's velocity map and Sersic radius,
and writes the training HDF5.

### System image (fast worker startup)

Loading the heavy Julia stack (Plots, PyCall, DataFrames, HDF5, ...) in
every worker is slow. `scripts/build_base_sysimage.jl` bakes those
dependencies into a portable base system image but leaves GalLearn,
UCITools, and Revise out, so editing those packages never invalidates the
image. Build it on a compute node under the juliaup Julia:

    julia --project=scripts scripts/build_base_sysimage.jl

Then run `transform_images.jl` with that image loaded (`-J`), and have the
run pass the image to every worker via `--sysimage`, so no worker
recompiles the baked dependencies.

### Submitting

Slurm submission scripts are cluster-specific and are kept out of the repo
(as are the machine-specific sbatch wrappers that build the system
image). A submission script for `transform_images.jl` should:

- pin `~/.juliaup/bin/julia` (the system image must load under the Julia
  that built it),
- rebuild the image if `Manifest.toml` or `build_base_sysimage.jl` is
  newer than it,
- cap the worker count with an `NWORKERS` variable, and
- run `transform_images.jl` with `-J <sysimage>`.

Memory note: each worker is a full Julia process (~2 GB), so `NWORKERS`
times that footprint must fit node RAM. Too many workers swap-thrash the
node; lower `NWORKERS` on memory-light nodes rather than raising `--mem`
past what the node has.

## Splitting the dataset

Training never sees a plain random split. A locked, stratified test set
holds out a fixed set of galaxies permanently, and every training run
consumes a separate, explicit train/val split file, so a run's exact
train/val galaxies are always traceable to one JSON file rather than a
seed.

Splitting keys galaxies by id, not by HDF5 row index, since
`src/Dataset.jl`'s `load_images` assembles its file list from unsorted
directory listings, so a given row index can point at a different
galaxy after a rebuild under the same filename. To catch that, lock
every dataset by content hash first:

    python scripts/lock_dataset.py --dataset <dataset filename>

This records the dataset file's sha256 under `dataset_hashes/` in the
repo. `gallearn.splitting` and `gallearn.train` refuse to run against a
dataset that has no lock yet, or whose current content no longer
matches its lock (`gallearn/dataset_lock.py`).

With the dataset locked, create or top up the locked test set:

    python scripts/split.py test-lock --dataset <dataset filename>

This draws galaxies into the test set with stratified sampling on
stellar mass and sSFR, so the locked set tracks the population's
mass/SFR distribution rather than a plain random draw. Locking is
append-only: rerunning it later, after the dataset has grown, only
ever adds newly eligible galaxies on top of the existing lock, never
reassigning or dropping one already locked. Each run writes a new,
immutable `splits/test_lock_v<N>.json` rather than editing the
previous version. The test lock excludes every projection of a locked
galaxy from training, not just the specific rows present when the
galaxy first entered the lock.

`test-lock` only finds something new to add when:

- the dataset contains galaxy ids it didn't contain before, or
- you raise `--test-fraction` above what's currently locked.

Rebuilding the dataset with only new projections of galaxies already
present, locked or not, gives `test-lock` nothing new to select:
`test-lock` skips a galaxy already in the lock, and for a galaxy not
yet locked, a previous run already counted it toward its stratum's
target, so the same `--test-fraction` won't pull it in either.

If you need more test rows, you need to expand image generation to
accept more viewing directions, Sersic-fitting
those new images (`scripts/gen_octant_shapes.py`), and rebuilding and
re-locking the dataset (`scripts/transform_images.jl`). Any such new
rows would land in test automatically, without a `test-lock` rerun,
since `scripts/split.py split` only pulls rows for the galaxies a
split's `train_galaxies`/`val_galaxies` lists actually name.

Then write a train/val split over whatever galaxies the test lock
excludes:

    python scripts/split.py split --dataset <dataset filename>

This writes `splits/split_<name>.json`. `--split-name` sets `<name>`
explicitly; omit it and `<name>` defaults to `<dataset filename's
stem>_v<N>`, N one more than the highest existing split version for
that dataset in `splits/`. A split is not stratified, and unlike the
test lock's single growing lineage, several splits are meant to be
simultaneously current: each is an independent random train/val
partition of the same non-locked galaxies, one per experiment, and
none supersedes another (rerun, or pass a different `--split-name`,
to get another one). Once written, a split file is immutable, since a
checkpoint or a resumed run may already reference it;
`scripts/split.py split` refuses to overwrite a split file that
already exists at the target name.

## Training the network

The network takes the image tensor (bands plus velocity map) and the
Sersic radius `Re` as a late-stage auxiliary input.

`scripts/train.py` is the unified training entry point. It parses the
command line and calls `gallearn.train.main()`, which selects the task
and architecture and covers all four combinations:

- `--task classifier` — star-forming vs quenched (BCE loss, F1 metric).
- `--task regressor` — sSFR on the star-forming subset (MSE loss, with
  asinh-scaled targets).
- `--model standard` — torchvision ResNet-18 backbone (`StandardNet`).
- `--model resnet` — the custom ResNet from `cnn.py`.

`--split <path>` names a train/val split JSON from `scripts/split.py
split` (see "Splitting the dataset" above) and is required on a fresh
run; there is no separate `--dataset` flag, since the split file's own
recorded `dataset_path` determines which dataset the run trains
against. A `--resume <checkpoint>` run reuses the checkpoint's own
recorded dataset, split, train/val row indices, task, model
architecture, and run name instead, and rejects `--split`, `--task`,
`--model`, and `--run-name` if any is also given, so a resumed run
can never silently continue as a different split, architecture, or
run than the one it started as. `--wandb` is rejected the same way:
a resumed run automatically continues its checkpoint's own wandb run
if it had one (by the run id recorded in the checkpoint), or
continues without wandb if it didn't, so there's no `--wandb r` mode
to forget and no way for a resumed run to silently drop a stretch of
metrics or land on the wrong chart.

`--pretrained` only affects `--model standard`: it starts the
ResNet-18 backbone from ImageNet weights instead of a random init.
`cnn.ResNet` has no pretrained option, since it isn't a standard
torchvision architecture with published weights.

Other options include `--epochs`, `--batch-size`, `--lr`, `--seed`,
`--wandb {n,y}`, `--run-name`, `--resume <checkpoint>`, and
`--no-scheduler`. It handles checkpoint save/resume, a
ReduceLROnPlateau scheduler, and optional Weights & Biases logging.

Run it as:

    python scripts/train.py --task classifier --model standard \
        --split splits/split_<name>.json

`gallearn/train.py` exposes the training logic as a plain, importable
`main()` with no argparse, so it can also be driven from a notebook or
REPL via `gallearn.train.main(...)`. `cnn.py` holds the model
definitions (`StandardNet`, `ResNet`, `BasicResBlock`) that `train.py`
imports. It also still carries an older standalone `main()` that
`scripts/run_cnn.py` launches; `gallearn/train.py` supersedes that
driver.

## Tests

`pytest` runs the Python test suite. `conftest.py` sets `GALLEARN_CONFIG`
to `.github/config_ci.ini`, so tests use the CI fixtures automatically
without any manual environment setup.
