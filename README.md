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
3. **Train** (Python). Load the HDF5 dataset and train the CNN.

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

## Training the network

The network takes the image tensor (bands plus velocity map) and the
Sersic radius `Re` as a late-stage auxiliary input.

`gallearn/train.py` is the unified training entry point. It selects the
task and architecture from the command line and covers all four
combinations:

- `--task classifier` — star-forming vs quenched (BCE loss, F1 metric).
- `--task regressor` — sSFR on the star-forming subset (MSE loss, with
  asinh-scaled targets).
- `--model bernoulli` — torchvision ResNet-18 backbone (`BernoulliNet`).
- `--model resnet` — the custom ResNet from `cnn.py`.

Other options include `--dataset`, `--epochs`, `--batch-size`, `--lr`,
`--seed`, `--wandb {n,y,r}`, `--run-name`, `--resume <checkpoint>`, and
`--disable-scheduler`. It handles checkpoint save/resume, a
ReduceLROnPlateau scheduler, and optional Weights & Biases logging.

Run it as a module, since it uses package-relative imports:

    python -m gallearn.train --task classifier --model bernoulli

`cnn.py` holds the model definitions (`BernoulliNet`, `ResNet`,
`BasicResBlock`) that `train.py` imports. It also still carries an older
standalone `main()` that the example `run_cnn.sh` launches; `train.py`
supersedes that driver. Training runs as a Slurm job.

## Tests

`pytest` runs the Python test suite. `conftest.py` sets `GALLEARN_CONFIG`
to `.github/config_ci.ini`, so tests use the CI fixtures automatically
without any manual environment setup.
