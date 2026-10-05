# Image format and axis conventions

A reference for how galaxy images are stored in the dataset and how to
display them correctly in matplotlib.

## Array layout

Images in the dataset HDF5 file are stored in NCHW order: batch
dimension first, then channels, then height, then width. The channel
order is r, g, u (the three photometric bands), followed by a fourth
vmap channel.

```
X[i, 0, :, :]  # r band, galaxy i
X[i, 1, :, :]  # g band
X[i, 2, :, :]  # u band
X[i, 3, :, :]  # velocity map
```

## Vertical axis convention

Row 0 is the top of the image (high-y in simulation space). Row index
increases downward. This matches matplotlib's default `origin='upper'`
convention.

The convention is set by `get_mock_observation` in
[`mockobservation_tools/galaxy_tools.py`](https://github.com/courtk32/mockobservation-tools/blob/main/mockobservation_tools/galaxy_tools.py). After projecting particles
onto a pixel grid, the function applies `np.rot90(..., k=1)` with the
comment "columns are x, row are y, first values is the upper left
value of the image." The result is that the stored arrays are already
in screen/image coordinates, not Cartesian coordinates.

**Consequence for `imshow`.** Call `ax.imshow(img)` without an
`origin` argument. Passing `origin='lower'` flips the image
vertically and produces an upside-down galaxy.

## Displaying an image

```python
import numpy as np

# img has shape (C, H, W); drop vmap, rearrange to (H, W, C)
rgb = np.stack([img[0], img[1], img[2]], axis=-1)
ax.imshow(rgb)  # no origin argument
```

`visual_checks.load_gal_for_imshow` does this via
`tensor.permute(0, 2, 3, 1)` on a batch tensor of shape (N, C, H, W),
producing (N, H, W, C), and then passes the result to `ax.imshow`
without an `origin` argument.
