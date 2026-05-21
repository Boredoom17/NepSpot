# Step 1: Download External Dataset (Beginner)

Goal: Put one dataset into the correct incoming folder and log it.

## What you will do now

1. Pick one source to start with: commonvoice.
2. Get the direct download URL for the archive.
3. Run one command.
4. Confirm file and registry entry are created.

## Command

From project root:

bash scripts/data_mining/01_download_dataset.sh \
  commonvoice \
  "PASTE_DIRECT_DOWNLOAD_URL_HERE" \
  "CC-0" \
  "your_name"

## Verify success

1. Check file exists under:
   - data/external/incoming/commonvoice/

2. Check registry row exists in:
   - data/external/manifests/source_registry.csv

## Important note

Do not download everything at once. Start with one source first (commonvoice), then we continue to Step 2.
