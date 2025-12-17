# External Raw Beamline Data (Not Tracked)

This directory is intentionally empty in the repository.

Raw BNB beam study CSV files must be supplied locally by the user, for example:

    bnb_br_study_2025-06-16/

## Recommended usage (symlink)

For convenience, it is recommended to create a symbolic link:

    ln -s ~/Downloads/bnb_br_study_2025-06-16 external_data/bnb_br_study_2025-06-16

This allows analysis scripts to be run using a stable relative path:

    --data-dir ../external_data/bnb_br_study_2025-06-16

The raw data itself is NOT tracked by git and is not redistributed.
