from pathlib import Path

from datasets.drivelm_dataset import DriveLMParser
from datasets.image_loader import MultiViewImageLoader
from datasets.sft_dataset import DriveLMSFTDataset


DATA_PATH = Path(
    "/home/leo/projects/DriveLM/challenge/data/train_sample.json"
)

IMAGE_ROOT = Path(
    "/home/leo/projects/DriveLM/"
    "challenge/llama_adapter_v2_multimodal7b/data/nuscenes"
)


def main():

    # --------------------------------------------
    # Canonical samples
    # --------------------------------------------

    parser = DriveLMParser(
        DATA_PATH
    )

    samples = parser.parse()

    # --------------------------------------------
    # Image loader
    # --------------------------------------------

    image_loader = MultiViewImageLoader(
        IMAGE_ROOT
    )

    # --------------------------------------------
    # SFT dataset
    # --------------------------------------------

    dataset = DriveLMSFTDataset(
        samples=samples,
        image_loader=image_loader,
    )

    print("=" * 60)
    print("DriveLM SFT Dataset Test")
    print("=" * 60)

    print(
        f"Dataset size: {len(dataset)}"
    )

    sample = dataset[0]

    print()
    print(
        "Sample ID:",
        sample["sample_id"],
    )

    print(
        "Task:",
        sample["task"],
    )

    print(
        "Question:",
        sample["question"],
    )

    print(
        "Answer:",
        sample["answer"],
    )

    print()
    print(
        "Number of images:",
        len(sample["images"]),
    )

    for i, image in enumerate(
        sample["images"]
    ):
        print(
            f"Image {i}: "
            f"{image.size} "
            f"{image.mode}"
        )

    print()
    print("Conversation")
    print("-" * 60)

    for message in sample["messages"]:

        print(
            f"\nRole: {message['role']}"
        )

        for content in message["content"]:

            if content["type"] == "text":

                print(
                    content["text"]
                )

            elif content["type"] == "image":

                print(
                    "<image>"
                )

    # --------------------------------------------
    # Basic assertions
    # --------------------------------------------

    assert len(dataset) == 790

    assert len(
        sample["images"]
    ) == 6

    assert (
        sample["messages"][0]["role"]
        == "user"
    )

    assert (
        sample["messages"][1]["role"]
        == "assistant"
    )

    assert (
        sample["answer"]
        ==
        sample["messages"][1]["content"][0]["text"]
    )

    print()
    print("=" * 60)
    print("SFT Dataset Test: PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()