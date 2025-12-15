#!/usr/bin/env bash

# Convert one or more .nsys-rep files into SQLite databases using nsys export.
set -euo pipefail
shopt -s nullglob

usage() {
  cat <<'EOF'
Usage: ./nsys-rep_to_sqlite.sh [report.nsys-rep ...]
Convert Nsight Systems reports (.nsys-rep) to SQLite databases (.sqlite).
If no files are given, every *.nsys-rep in the current directory is converted.
EOF
}

if ! command -v nsys >/dev/null 2>&1; then
  echo "Error: nsys not found in PATH." >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  reps=( *.nsys-rep )
  if [[ ${#reps[@]} -eq 0 ]]; then
    usage
    exit 1
  fi
else
  reps=( "$@" )
fi

for rep in "${reps[@]}"; do
  if [[ ! -f "$rep" ]]; then
    echo "Warning: '$rep' not found; skipping." >&2
    continue
  fi

  dir=$(dirname -- "$rep")
  file=$(basename -- "$rep")
  if [[ "$file" == *.nsys-rep ]]; then
    stem=${file%.nsys-rep}
  else
    stem=$file
  fi

  output="${dir}/${stem}.sqlite"
  echo "Exporting ${rep} -> ${output}"
  nsys export --type sqlite --force-overwrite=true --output "$output" "$rep"
done

echo "Conversion complete."
