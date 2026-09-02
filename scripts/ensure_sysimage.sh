#!/bin/bash
# Build the base sysimage only when something that changes its contents
# changed.
#
# Rebuilding takes tens of minutes: PackageCompiler compiles the whole
# dependency stack ahead of time for three CPU targets and links a
# ~1.4 GB shared library, with none of the per-package incremental reuse
# ordinary precompilation gets. So the check for whether to rebuild
# should fire when the image would actually differ, and stay quiet
# otherwise.
#
# Comparing modification times cannot tell those apart. Editing a
# comment in build_base_sysimage.jl, checking the file out again, or
# copying the tree all move an mtime forward without changing a byte the
# image depends on, and each one costs a full rebuild.
#
# So hash the inputs that determine the image instead:
#
#   - Manifest.toml, which pins the exact dependency versions baked in.
#   - build_base_sysimage.jl's code, minus whole-line comments and blank
#     lines, which carries BASE_PACKAGES and the default CPU_TARGET.
#   - JULIA_CPU_TARGET, which overrides that default when set.
#   - The Julia version, since an image only loads under the Julia that
#     built it.
#
# The hash goes in a stamp file beside the image. A missing image, a
# missing stamp, or a differing stamp triggers a rebuild. A trailing
# comment on a line of code still counts toward the hash, which can
# cause one unnecessary rebuild; that errs toward rebuilding, which is
# the safe direction.
#
# Usage, from the scripts/ directory:
#
#     ./ensure_sysimage.sh "$JULIA" "$SYSIMAGE"
set -euo pipefail

JULIA="$1"
SYSIMAGE="$2"
STAMP="${SYSIMAGE}.stamp"
RECIPE="./build_base_sysimage.jl"
MANIFEST="./Manifest.toml"

want=$(
    {
        cat "$MANIFEST"
        grep -vE '^[[:space:]]*(#|$)' "$RECIPE"
        echo "cpu_target=${JULIA_CPU_TARGET:-}"
        "$JULIA" --version
    } | sha256sum | cut -d' ' -f1
)

have=""
if [ -f "$STAMP" ]; then
    have=$(cat "$STAMP")
fi

if [ -f "$SYSIMAGE" ] && [ "$want" = "$have" ]; then
    echo "Base sysimage is current (${want:0:12}); skipping rebuild."
    exit 0
fi

if [ ! -f "$SYSIMAGE" ]; then
    echo "Building base sysimage: none at $SYSIMAGE."
elif [ -z "$have" ]; then
    echo "Building base sysimage: no stamp, so its inputs are unknown."
else
    echo "Building base sysimage: inputs changed" \
        "(${have:0:12} -> ${want:0:12})."
fi

"$JULIA" --project=./ "$RECIPE"

# Stamp only after a successful build, so an interrupted or failed one
# leaves the image looking stale and gets retried rather than skipped.
echo "$want" > "$STAMP"
echo "Wrote sysimage stamp ${want:0:12}."
