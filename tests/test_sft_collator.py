# tests/test_sft_collator.py

from pathlib import Path

import torch

from transformers import AutoProcessor

from datasets.drivelm_dataset import DriveLMParser
from datasets.image_loader import MultiViewImageLoader
from datasets.sft_dataset import DriveLMSFTDataset
from datasets.sft_collator import Qwen3VLSFTCollator


MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"

DATA_PATH = Path(
    "/home/leo/projects/DriveLM/"
    "challenge/data/train_sample.json"
)

IMAGE_ROOT = Path(
    "/home/leo/projects/DriveLM/"
    "challenge/llama_adapter_v2_multimodal7b/data/nuscenes"
)


def main():

    print("=" * 60)
    print("Qwen3-VL SFT Collator Test")
    print("=" * 60)

    # ---------------------------------------
    # 1. Load canonical dataset
    # ---------------------------------------

    parser = DriveLMParser(DATA_PATH)

    samples = parser.parse()

    image_loader = MultiViewImageLoader(
        IMAGE_ROOT
    )

    dataset = DriveLMSFTDataset(
        samples=samples,
        image_loader=image_loader,
    )

    sample = dataset[0]

    # ---------------------------------------
    # 2. Load processor
    #
    # Processor only, not model weights.
    # ---------------------------------------

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    # ---------------------------------------
    # 3. Same visual budget as baseline
    # ---------------------------------------

    visual_tokens_per_image = 256

    max_pixels = (
        visual_tokens_per_image * 32 * 32
    )

    min_pixels = 64 * 32 * 32

    processor.image_processor.size = {
        "longest_edge": max_pixels,
        "shortest_edge": min_pixels,
    }

    # ---------------------------------------
    # 4. Create collator
    # ---------------------------------------

    collator = Qwen3VLSFTCollator(
        processor=processor
    )

    batch = collator([sample])

    # ---------------------------------------
    # 5. Inspect model inputs
    # ---------------------------------------

    print("\nModel Inputs")
    print("-" * 60)

    for key, value in batch.items():

        if isinstance(value, torch.Tensor):

            print(
                f"{key:25s}: "
                f"{tuple(value.shape)}"
            )

    # ---------------------------------------
    # 6. Inspect labels
    # ---------------------------------------

    input_ids = batch["input_ids"]

    labels = batch["labels"]

    masked_positions = (
        labels == -100
    )

    supervised_positions = (
        labels != -100
    )

    num_masked = (
        masked_positions.sum().item()
    )

    num_supervised = (
        supervised_positions.sum().item()
    )

    print("\nLabel Statistics")
    print("-" * 60)

    print(
        "Total tokens:",
        input_ids.numel()
    )

    print(
        "Masked tokens:",
        num_masked
    )

    print(
        "Supervised tokens:",
        num_supervised
    )

    # ---------------------------------------
    # 7. Decode supervised tokens
    # ---------------------------------------

    supervised_ids = input_ids[
        supervised_positions
    ]

    supervised_text = (
        processor.tokenizer.decode(
            supervised_ids.tolist(),
            skip_special_tokens=False,
        )
    )

    print("\nSupervised Text")
    print("-" * 60)

    print(supervised_text)

    # ---------------------------------------
    # 8. Assertions
    # ---------------------------------------

    assert input_ids.shape == labels.shape

    assert num_masked > 0

    assert num_supervised > 0

    # The supervised suffix should contain GT.
    assert (
        sample["answer"]
        in supervised_text
    )

    # First supervised position marks the
    # beginning of the assistant suffix.
    first_supervised = (
        supervised_positions[0]
        .nonzero(as_tuple=True)[0][0]
        .item()
    )

    assert torch.all(
        labels[0, :first_supervised] == -100
    )

    assert torch.equal(
        labels[0, first_supervised:],
        input_ids[0, first_supervised:],
    )

    # ---------------------------------------
    # 9. Visual-token statistics
    # ---------------------------------------

    if "image_grid_thw" in batch:

        grids = batch["image_grid_thw"]

        assert grids.shape[0] == 6

        merge_size = (
            processor.image_processor
            .merge_size
        )

        total_visual_tokens = (
            grids.prod(dim=1)
            // (merge_size ** 2)
        ).sum().item()

        print("\nVisual Tokens")
        print("-" * 60)

        print(
            "Image grids:",
            grids.tolist()
        )

        print(
            "Total visual tokens:",
            total_visual_tokens
        )

        assert total_visual_tokens == 1512

    print()
    print("=" * 60)
    print("SFT Collator Test: PASS")
    print("=" * 60)

    print("\nAssistant Token Inspection")
    print("-" * 60)

    for token_id in supervised_ids.tolist():

        token_text = processor.tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
        )

        print(
            f"ID: {token_id:8d} "
            f"Token: {repr(token_text)}"
        )

if __name__ == "__main__":
    main()