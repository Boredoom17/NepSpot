# Daily Mining Checklist

Use this every day while collecting clips.

## Before You Start

- Run: bash scripts/data_mining/make_dataset_scaffold.sh
- Open manifests in data/external/manifests/.
- Pick one source only (Common Voice or OpenSLR or YouTube) for this session.

## During Mining

- Log source in source_registry.csv.
- Add every candidate to candidates.csv.
- Keep clip IDs unique.
- Keep speaker IDs consistent.

## During Review

- Listen to each candidate once with headphones.
- Accept only if keyword is clear and centered.
- Move decision to accepted.csv or rejected.csv.
- Write rejection reason.

## End of Day

- Check accepted counts by keyword.
- Check how many unique speakers were added.
- Note sparse keywords for next session.
- Backup manifests.
