from __future__ import annotations

from collections.abc import Iterable

from slime.utils.types import Sample


def _flatten_samples(samples: list[Sample] | list[list[Sample]]) -> list[Sample]:
    if not samples:
        return []
    if isinstance(samples[0], Sample):
        return samples  # type: ignore[return-value]

    flat_samples: list[Sample] = []
    for group in samples:  # type: ignore[assignment]
        if isinstance(group, Iterable):
            flat_samples.extend(group)
        else:
            raise TypeError(f"Expected list[Sample] entries, but got: {type(group)}")
    return flat_samples


def _split_lengths(sample: Sample) -> list[int]:
    archived_segments = getattr(sample, "context_reset_token_segments", None) or []
    split_lengths = [len(segment) for segment in archived_segments if segment]

    remaining_length = sample.response_length - sum(split_lengths)
    if remaining_length < 0:
        raise ValueError(
            "Invalid `context_reset_token_segments`: segment lengths exceed response_length. "
            f"response_length={sample.response_length}, split_lengths={split_lengths}"
        )
    if remaining_length > 0 or not split_lengths:
        split_lengths.append(remaining_length)

    return [length for length in split_lengths if length > 0]


def convert_samples_to_train_data(args, samples: list[Sample] | list[list[Sample]]) -> dict:
    """
    Expand each sample into multiple train items by splitting response tokens at context reset boundaries.

    This function is designed for trajectories produced by `generate_with_bash_retool.py`, where
    `sample.context_reset_token_segments` stores response chunks that were dropped from model context.
    """
    flat_samples = _flatten_samples(samples)
    if not flat_samples:
        return {
            "tokens": [],
            "response_lengths": [],
            "rewards": [],
            "raw_reward": [],
            "truncated": [],
            "sample_indices": [],
            "loss_masks": [],
        }

    expanded = {
        "tokens": [],
        "response_lengths": [],
        "raw_reward": [],
        "truncated": [],
        "sample_indices": [],
        "loss_masks": [],
    }

    optional_fields = {
        "round_number": [],
        "rollout_log_probs": [],
        "rollout_routed_experts": [],
        "metadata": [],
        "multimodal_train_inputs": [],
        "teacher_log_probs": [],
    }

    for sample in flat_samples:
        split_lengths = _split_lengths(sample)
        if not split_lengths:
            continue

        reward_value = sample.get_reward_value(args)
        split_reward = reward_value / len(split_lengths)

        prompt_length = len(sample.tokens) - sample.response_length
        prompt_tokens = sample.tokens[:prompt_length]
        response_tokens = sample.tokens[prompt_length:]

        if sample.loss_mask is None:
            sample.loss_mask = [1] * sample.response_length
        if len(sample.loss_mask) != sample.response_length:
            raise ValueError(
                f"loss_mask length {len(sample.loss_mask)} != response_length {sample.response_length}"
            )

        offset = 0
        for piece_idx, piece_length in enumerate(split_lengths):
            piece_response_tokens = response_tokens[offset : offset + piece_length]
            piece_loss_mask = sample.loss_mask[offset : offset + piece_length]
            piece_rollout_log_probs = (
                sample.rollout_log_probs[offset : offset + piece_length] if sample.rollout_log_probs is not None else None
            )
            piece_teacher_log_probs = (
                sample.teacher_log_probs[offset : offset + piece_length] if sample.teacher_log_probs is not None else None
            )
            offset += piece_length

            if sample.remove_sample:
                piece_loss_mask = [0] * piece_length

            expanded["tokens"].append(prompt_tokens + piece_response_tokens)
            expanded["response_lengths"].append(piece_length)
            expanded["raw_reward"].append(split_reward)
            expanded["truncated"].append(
                1 if piece_idx == len(split_lengths) - 1 and sample.status == Sample.Status.TRUNCATED else 0
            )
            expanded["sample_indices"].append(sample.index)
            expanded["loss_masks"].append(piece_loss_mask)

            if sample.metadata and "round_number" in sample.metadata:
                optional_fields["round_number"].append(sample.metadata["round_number"])
            if piece_rollout_log_probs is not None:
                optional_fields["rollout_log_probs"].append(piece_rollout_log_probs)
            if sample.rollout_routed_experts is not None:
                optional_fields["rollout_routed_experts"].append(sample.rollout_routed_experts)
            if sample.train_metadata is not None:
                metadata = dict(sample.train_metadata)
                metadata["piece_index"] = piece_idx
                metadata["num_pieces"] = len(split_lengths)
                optional_fields["metadata"].append(metadata)
            if sample.multimodal_train_inputs is not None:
                optional_fields["multimodal_train_inputs"].append(sample.multimodal_train_inputs)
            if piece_teacher_log_probs is not None:
                optional_fields["teacher_log_probs"].append(piece_teacher_log_probs)

        if offset != sample.response_length:
            raise ValueError(
                f"Split lengths ({split_lengths}) do not cover full response_length ({sample.response_length})."
            )

    # Use split rewards directly. Group-wise normalization is skipped intentionally because
    # each original sample can expand into a different number of pieces.
    expanded["rewards"] = list(expanded["raw_reward"])

    if flat_samples[0].metadata and "raw_reward" in flat_samples[0].metadata:
        # Preserve external raw_reward override while still dividing by number of pieces.
        overridden_raw_rewards: list[float] = []
        for sample in flat_samples:
            split_lengths = _split_lengths(sample)
            if not split_lengths:
                continue
            raw_reward = sample.metadata["raw_reward"] / len(split_lengths)
            overridden_raw_rewards.extend([raw_reward] * len(split_lengths))
        expanded["raw_reward"] = overridden_raw_rewards
        expanded["rewards"] = list(overridden_raw_rewards)

    for key, values in optional_fields.items():
        if values:
            expanded[key] = values

    return expanded
