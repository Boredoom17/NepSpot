# Step 0: Ground Rules (Read Once)

## Can you skip voice checking?

Short answer: no.

You can reduce manual checks, but you should not skip them completely.

Why:

- Automatic transcript matching can be wrong.
- Keyword timing can be wrong.
- Noise or music can dominate a clip.
- Wrong clips hurt model quality a lot.

## Minimum safe review policy

- Manually listen to at least 20% of candidate clips per keyword.
- Manually review 100% for rare keywords.
- Reject any clip where keyword is unclear.

## About creating/downloading datasets

- You can download open datasets (Common Voice, OpenSLR).
- You should not create fake speech as replacement for real speakers.
- Augmentation is useful, but it does not replace speaker diversity.
