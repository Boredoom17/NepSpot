# Step 2: Extract & Preview Dataset

Goal: Unpack the Common Voice tar.gz and show what's inside.

## Command

From project root:

bash scripts/data_mining/02_extract_and_find_keywords.sh commonvoice

## What this does

1. Extracts the tar.gz file to: data/external/staging/extracted/commonvoice/
2. Finds the clips.tsv file (Common Voice metadata).
3. Shows first few lines so you can see the structure.
4. Takes about 1-2 minutes.

## Then what?

After you run this command, reply "step 3" and I will show you Step 3 (which extracts clips with keywords).
