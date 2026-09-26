"""Independent, reproducible RNG streams for pre-repeated validation requests."""

from copy import deepcopy
from hashlib import blake2b
from numbers import Integral


def validation_request_seed(base_seed, step, question_id, sample_index):
    """Derive a nonnegative int64 seed independent of DP rank and microbatching."""
    values = (base_seed, step, question_id, sample_index)
    if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in values):
        raise ValueError("Validation seed coordinates must be nonnegative integers")
    identity = ":".join(str(int(value)) for value in values).encode("ascii")
    return int.from_bytes(blake2b(identity, digest_size=8, person=b"verl-val-seeds").digest(), "little") & (
        (1 << 63) - 1
    )


def per_request_sampling_params(base_params, seeds, *, expected_rows):
    """Clone native-n=1 params per row; never mutate training/teacher defaults.

    vLLM seeds every independent request from SamplingParams.seed. Repeating a
    prompt 32 times with the same seed produces duplicated samples, unlike its
    native n=32 path that increments each child seed. Supply distinct seeds for
    the pre-repeated validation rows instead.
    """
    if base_params.n != 1:
        raise ValueError("Per-row validation seeds require native SamplingParams.n=1")
    if len(seeds) != expected_rows:
        raise ValueError("Validation seed count must match the number of input rows")
    output = []
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < (1 << 63):
            raise ValueError("Each validation seed must be a nonnegative int64")
        params = deepcopy(base_params)
        params.seed = int(seed)
        output.append(params)
    return output
