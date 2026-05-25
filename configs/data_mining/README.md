# Data Mining Scripts

This folder contains helper scripts for preparing and tracking external keyword mining data.

## Quick Start

1. From project root, run:

   bash scripts/data_mining/make_dataset_scaffold.sh

2. Fill manifest files in:

   data/external/manifests/

3. Keep incoming source files separated by source:

   - data/external/incoming/commonvoice/
   - data/external/incoming/openslr/
   - data/external/incoming/youtube/

4. Put auto-cut clips into:

   - data/external/staging/raw_clips/

5. After manual review:

   - accepted clips metadata -> data/external/manifests/accepted.csv
   - rejected clips metadata -> data/external/manifests/rejected.csv

6. Move accepted, normalized clips to:

   - data/external/staging/for_nepspot_raw/

7. Copy accepted clips into NepSpot training layout:

   data/raw/<speaker_id>/<keyword>/<clip>.wav

## Important Rules

- Keep existing validation and test speakers unchanged.
- Add all new external speakers to train only.
- Do not publish raw YouTube clips in any public dataset release.
- Always log source URL/reference in source_registry.csv.
