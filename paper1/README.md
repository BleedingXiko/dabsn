# Paper 1 build

This directory is the selected public paper-1 source bundle. It contains the
TeX source, bibliography, checked-in bibliography output, three final figures,
and only the machine-readable result tables used by the paper.

Build with `make`. The target uses Tectonic against the checked-in source and
bibliography; it does not require access to the source workspace.

The reproduction programs are packaged under `dabsn.reproductions` and exposed
through the `dabsn-reproduce-*` commands at repository root. Their full defaults
match the verified configurations recorded in the included CSVs; reduced
settings are supported for executable smoke tests.

Regenerate all three figures with `python generate_figures.py` after installing
the `paper` optional dependency.
