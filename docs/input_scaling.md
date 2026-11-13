# Input scaling

This document covers how the pipeline transforms the network's inputs, the four-channel image tensor and the Sersic radius `Re`, before they reach the model. See [target scaling](target_scaling.md) for how the pipeline scales the targets.

## Where scaling happens

`preprocessing.LazyGalaxyDataset.__getitem__` scales each image as the DataLoader reads it from the HDF5 file, one sample at a time. The models themselves never scale anything. `cnn.StandardNet` and `cnn.ResNet` expect input that the dataset has already scaled. Training (`train.main`) and evaluation (`scripts/evaluate.py`) both build their datasets from this class, so both apply the same transform.

## The per-channel transform

The image tensor holds r, g, and u in channels 0 to 2 and the velocity map in channel 3 (see [image format](image_format.md)). `__getitem__` applies three steps in order.

1. **Bands (channels 0 to 2).** `asinh(stretch * x)` with `stretch = 1e-5`. Asinh compresses bright pixels roughly logarithmically, stays nearly linear near zero, and stays defined at zero and below, so it needs no special handling for empty or negative pixels.
2. **Velocity map (channel 3).** NaN becomes 0. `UCITools` writes NaN wherever the gas column density falls below `min_cden`, so this step gives every gas-free pixel a finite value. The vmap gets no asinh.
3. **All four channels.** `(x - mean_c) / std_c`, using per-channel statistics fit on the training set.

The order has a visible consequence. Step 2 runs before step 3, so a gas-free vmap pixel reaches the network as `-mean_3 / std_3`, not as zero.

## Where the statistics come from

`preprocessing.compute_scaling_stats(hdf5_path, train_idxs)` computes the per-channel means and standard deviations. It makes two chunked passes over the HDF5 file, 256 rows at a time, so it never loads the full image tensor. Each pass applies steps 1 and 2 above, then the first pass accumulates means and the second accumulates squared deviations from them. It averages over every pixel of every training image, accumulates in float64, and returns the population standard deviation.

It reads only `train_idxs`, so validation and locked-test rows never inform the statistics. At that point `train_idxs` holds the split file's training galaxies narrowed to the rows valid for the task (the regressor, for example, drops quenched rows). The vmap statistics include every gas-free pixel, which step 2 has already set to zero.

`train.main` computes the statistics before it applies `train_orientations`. A run that restricts training to a subset of orientations therefore still scales with statistics fit on all orientations of its training galaxies.

## Reuse across resume and evaluation

`train.save_checkpoint` stores the statistics in every checkpoint as `scaling_means` and `scaling_stds`, along with the stretch they were fit on as `scaling_stretch`. A resumed run reuses all three verbatim instead of refitting, and `scripts/evaluate.py` builds its validation dataset from the same checkpoint values. A model therefore always sees inputs scaled exactly as they were during its own training.

## The stretch constant

`gallearn.preprocessing.IMAGE_STRETCH` sets the band stretch for new training runs. A fresh run passes it explicitly to both `compute_scaling_stats` and `LazyGalaxyDataset`, so the statistics and the dataset always agree, and saves it in the checkpoint. `compute_scaling_stats`, `sasinh_imgs_sscale_vmaps`, and `std_asinh` also use it as their default `stretch`.

Once a model exists, its checkpoint decides the stretch, not the constant. Resumed runs and `scripts/evaluate.py` read `scaling_stretch` back from the checkpoint. Checkpoints saved before that key existed fall back to a literal `1e-5`, the value every one of them used, so changing `IMAGE_STRETCH` later never changes how an existing model scales its inputs.

## Ablation channels

`LazyGalaxyDataset` also takes `zero_channels` and `noise_channels` for input ablations, and it applies both after scaling.

- A zeroed channel reaches the network as exactly 0, which in standardized units equals that channel's training mean.
- A noise channel gets fresh standard-normal noise on every read, which matches the scale of a standardized channel and gives a galaxy a different field each epoch.

## Re

`Re` gets no scaling at all. `preprocessing.load_metadata` reads it from the HDF5 file as stored, `LazyGalaxyDataset` returns it alongside each image, and both models concatenate it onto the pooled feature vector just before the head. `cnn.ResNet` applies its pre-head dropout before the concatenation, so `Re` always survives. The head's first linear layer sees `Re` at its native magnitude, in whatever units the HDF5 file stores.

## The legacy whole-array function

`preprocessing.sasinh_imgs_sscale_vmaps` applies the same three steps to a full in-memory array, with two differences. It fits the statistics on whatever array it receives, which typically includes validation rows, and its standardization (`preprocessing.std_scale`) uses torch's unbiased standard deviation rather than the population value. Training does not use it. `notebooks/verify_X_to_vmaps.ipynb` and `tests/test_StandardNet.py` do.
