#!/usr/bin/env bash

set -euo pipefail

# Sort by the number inside "(N)"
mapfile -t files < <(
    printf '%s\n' files\ \(*\).zip |
    sed -E 's/files \(([0-9]+)\)\.zip/\1 &/' |
    sort -n |
    cut -d' ' -f2-
)

n=1

for f in "${files[@]}"; do
    new="batch-${n}.zip"

    echo "Renaming: $f -> $new"
    mv "$f" "$new"

    dir="batch-${n}"
    mkdir -p "$dir"

    echo "Extracting: $new -> $dir/"
    unzip -q "$new" -d "$dir"

    ((n++))
done

echo "Done."
