"""Phase 1 augmentation pipeline (deterministic).

Every random draw uses tf.random.stateless_* with a seed pair derived from a
per-example or per-batch index obtained via Dataset.enumerate(). Combined with
TF_DETERMINISTIC_OPS=1, tf.config.experimental.enable_op_determinism(), seeded
shuffle, num_parallel_calls=1, deterministic=True, and prefetch(1), two runs
of the same training script with the same dataset must produce identical
augmentation outcomes and identical INT8 test accuracy.

Pipeline layout:
    base_ds = from_tensor_slices((X, y))
            .shuffle(buf, seed=42, reshuffle_each_iteration=True)
            .enumerate()                          # per-example index
            .map(specaug, num_parallel_calls=1, deterministic=True)
            .batch(N)
            .enumerate()                          # per-batch index
            .map(mixup, num_parallel_calls=1, deterministic=True)
            .prefetch(1)
"""

import os

import numpy as np
import tensorflow as tf


_SEED = int(os.environ.get('SEED', '42'))
_BASE_SEED = tf.constant(_SEED, dtype=tf.int64)


def _mask_along_axis_stateless(x, axis, min_width, max_width, dim_size, seed_w, seed_s):
    width = tf.random.stateless_uniform(
        [], seed=seed_w, minval=min_width, maxval=max_width + 1, dtype=tf.int32,
    )
    start = tf.random.stateless_uniform(
        [], seed=seed_s, minval=0, maxval=dim_size - width + 1, dtype=tf.int32,
    )

    idx = tf.range(dim_size)
    keep = tf.logical_or(idx < start, idx >= start + width)

    if axis == 'freq':
        keep_mask = tf.reshape(keep, [dim_size, 1, 1])
        masked_count = width * tf.shape(x)[1] * tf.shape(x)[2]
    else:
        keep_mask = tf.reshape(keep, [1, dim_size, 1])
        masked_count = width * tf.shape(x)[0] * tf.shape(x)[2]

    x_masked = x * tf.cast(keep_mask, x.dtype)
    return x_masked, masked_count


def _specaugment_example(idx, x, y):
    """Deterministic SpecAugment. `idx` is the per-example position from .enumerate()."""
    idx_i = tf.cast(idx, tf.int64)
    seed_apply = tf.stack([idx_i, _BASE_SEED])
    apply_aug = tf.random.stateless_uniform([], seed=seed_apply) < 0.7

    def _apply():
        x_aug = x
        masked_total = tf.constant(0, dtype=tf.int32)

        # 2 time masks (width 1-3 on time dim 32)
        seed_w = tf.stack([idx_i, _BASE_SEED + 100])
        seed_s = tf.stack([idx_i, _BASE_SEED + 200])
        x_aug, m = _mask_along_axis_stateless(x_aug, 'time', 1, 3, 32, seed_w, seed_s)
        masked_total += m

        seed_w = tf.stack([idx_i, _BASE_SEED + 101])
        seed_s = tf.stack([idx_i, _BASE_SEED + 201])
        x_aug, m = _mask_along_axis_stateless(x_aug, 'time', 1, 3, 32, seed_w, seed_s)
        masked_total += m

        # 2 freq masks (width 2-5 on freq dim 40)
        seed_w = tf.stack([idx_i, _BASE_SEED + 300])
        seed_s = tf.stack([idx_i, _BASE_SEED + 400])
        x_aug, m = _mask_along_axis_stateless(x_aug, 'freq', 2, 5, 40, seed_w, seed_s)
        masked_total += m

        seed_w = tf.stack([idx_i, _BASE_SEED + 301])
        seed_s = tf.stack([idx_i, _BASE_SEED + 401])
        x_aug, m = _mask_along_axis_stateless(x_aug, 'freq', 2, 5, 40, seed_w, seed_s)
        masked_total += m

        return x_aug, y, tf.constant(True), masked_total

    def _skip():
        return x, y, tf.constant(False), tf.constant(0, dtype=tf.int32)

    return tf.cond(apply_aug, _apply, _skip)


def _mixup_batch(batch_idx, x_batch, y_batch, spec_flags, masked_counts):
    """Deterministic Mixup. `batch_idx` is the per-batch index from .enumerate()."""
    bidx = tf.cast(batch_idx, tf.int64)
    seed_apply = tf.stack([bidx, _BASE_SEED + 1000])
    apply_mix = tf.random.stateless_uniform([], seed=seed_apply) < 0.5

    def _apply():
        seed_a = tf.stack([bidx, _BASE_SEED + 2000])
        seed_b = tf.stack([bidx, _BASE_SEED + 3000])
        gamma_a = tf.random.stateless_gamma([], seed=seed_a, alpha=0.2)
        gamma_b = tf.random.stateless_gamma([], seed=seed_b, alpha=0.2)
        lam = gamma_a / (gamma_a + gamma_b)

        # Deterministic permutation: argsort of stateless uniform scores.
        # stable=True guarantees ordering on the (vanishingly unlikely) tie.
        n = tf.shape(x_batch)[0]
        seed_perm = tf.stack([bidx, _BASE_SEED + 4000])
        scores = tf.random.stateless_uniform([n], seed=seed_perm)
        indices = tf.argsort(scores, stable=True)
        x_other = tf.gather(x_batch, indices)
        y_other = tf.gather(y_batch, indices)

        lam_x = tf.cast(lam, x_batch.dtype)
        lam_y = tf.cast(lam, y_batch.dtype)

        mixed_x = lam_x * x_batch + (1.0 - lam_x) * x_other
        mixed_y = lam_y * y_batch + (1.0 - lam_y) * y_other
        return mixed_x, mixed_y, spec_flags, masked_counts, tf.constant(True)

    def _skip():
        return x_batch, y_batch, spec_flags, masked_counts, tf.constant(False)

    return tf.cond(apply_mix, _apply, _skip)


def build_phase1_train_datasets(X_train, y_train_onehot, batch_size=32, shuffle_buffer=2048):
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train_onehot = np.asarray(y_train_onehot, dtype=np.float32)

    base_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_onehot))
    base_ds = base_ds.shuffle(
        min(shuffle_buffer, len(X_train)),
        seed=_SEED,
        reshuffle_each_iteration=True,
    )

    enumerated_ds = base_ds.enumerate()
    spec_ds = enumerated_ds.map(
        lambda idx, xy: _specaugment_example(idx, xy[0], xy[1]),
        num_parallel_calls=1,
        deterministic=True,
    )

    batched_ds = spec_ds.batch(batch_size, drop_remainder=False)
    batched_enum_ds = batched_ds.enumerate()
    debug_ds = batched_enum_ds.map(
        lambda bidx, batch: _mixup_batch(bidx, batch[0], batch[1], batch[2], batch[3]),
        num_parallel_calls=1,
        deterministic=True,
    )
    debug_ds = debug_ds.prefetch(1)

    train_ds = debug_ds.map(
        lambda x, y, _spec, _masked, _mix: (x, y),
        num_parallel_calls=1,
        deterministic=True,
    )
    train_ds = train_ds.prefetch(1)

    return train_ds, debug_ds


def inspect_augmentation_behavior(debug_ds, max_batches=12):
    total_examples = 0
    specaug_examples = 0
    zero_mask_examples = 0
    mixup_batches = 0
    observed_batches = 0

    for x_batch, _y_batch, spec_flags, masked_counts, mix_flag in debug_ds.take(max_batches):
        observed_batches += 1
        batch_size = int(x_batch.shape[0])
        total_examples += batch_size

        spec_np = spec_flags.numpy().astype(bool)
        masked_np = masked_counts.numpy()

        specaug_examples += int(np.sum(spec_np))
        zero_mask_examples += int(np.sum(masked_np > 0))
        mixup_batches += int(bool(mix_flag.numpy()))

    return {
        'observed_batches': observed_batches,
        'total_examples': total_examples,
        'specaug_examples': specaug_examples,
        'zero_mask_examples': zero_mask_examples,
        'mixup_batches': mixup_batches,
    }
