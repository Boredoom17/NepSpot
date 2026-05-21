# NepSpot Dataset Expansion Workspace

This folder documents the beginner-friendly pipeline for Option 4:

- In-house NepSpot speakers (existing)
- Mozilla Common Voice mining
- OpenSLR mining
- Targeted YouTube mining for sparse keywords

## Goals

- Reach 100+ effective speakers without changing evaluation fairness.
- Keep train/val/test protocol unchanged for apples-to-apples comparisons.
- Preserve legal/provenance metadata for every external clip.

## Files You Should Read First

- step-00-rules.md
- step-01-download.md
- step-by-step-beginner.md
- ../../configs/data_mining/source_plan.yaml
- ../../configs/data_mining/keyword_aliases.json

## What This Adds To Your Project

- A clean data/external/ workflow separate from data/raw and data/processed.
- Manifest templates for candidates, accepted clips, rejected clips, and speaker/source registries.
- A shell script to create or repair the data-mining scaffold in one command.
