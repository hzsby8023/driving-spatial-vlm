# datasets/sft_collator.py

import torch


class Qwen3VLSFTCollator:
    """
    Qwen3-VL multimodal SFT collator.

    V0:
        - batch_size = 1
        - six-view images
        - assistant-only loss
        - no padding
        - explicit prefix alignment check

    This version prioritizes correctness.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):

        # ---------------------------------------
        # V0: only batch size 1
        # ---------------------------------------

        if len(features) != 1:
            raise ValueError(
                "Qwen3VLSFTCollator V0 "
                "requires batch_size=1."
            )

        sample = features[0]

        messages = sample["messages"]

        # ---------------------------------------
        # 1. Full conversation:
        #
        # user + assistant ground truth
        # ---------------------------------------

        full_inputs = (
            self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )
        )

        # ---------------------------------------
        # 2. Prefix:
        #
        # user + assistant header
        #
        # No assistant answer.
        # ---------------------------------------

        prefix_messages = messages[:-1]

        prefix_inputs = (
            self.processor.apply_chat_template(
                prefix_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        )

        # ---------------------------------------
        # 3. Get token IDs
        # ---------------------------------------

        input_ids = full_inputs["input_ids"]

        prefix_ids = prefix_inputs["input_ids"]

        full_length = input_ids.shape[1]

        prefix_length = prefix_ids.shape[1]

        if prefix_length >= full_length:
            raise RuntimeError(
                "Invalid SFT sequence lengths: "
                f"prefix={prefix_length}, "
                f"full={full_length}"
            )

        # ---------------------------------------
        # 4. Verify token alignment
        # ---------------------------------------

        actual_prefix = input_ids[
            :,
            :prefix_length
        ]

        if not torch.equal(
            actual_prefix,
            prefix_ids,
        ):
            raise RuntimeError(
                "Multimodal prompt prefix mismatch. "
                "Check chat template and processor."
            )

        # ---------------------------------------
        # 5. Build labels
        # ---------------------------------------

        labels = input_ids.clone()

        # Mask user / image / prompt tokens
        labels[
            :,
            :prefix_length
        ] = -100

        # Remaining tokens:
        # assistant GT + end-of-turn marker

        # ---------------------------------------
        # 6. Build model batch
        # ---------------------------------------

        batch = dict(full_inputs)

        batch["labels"] = labels

        return batch