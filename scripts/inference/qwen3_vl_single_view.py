import torch

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
)

from datasets.drivelm_dataset import DriveLMParser
from datasets.image_loader import MultiViewImageLoader


MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

ANNOTATION_PATH = (
    "/home/leo/projects/DriveLM/"
    "challenge/data/train_sample.json"
)

NUSCENES_ROOT = (
    "/home/leo/projects/DriveLM/"
    "challenge/llama_adapter_v2_multimodal7b/"
    "data/nuscenes"
)


def main():
    # --------------------------------------------------
    # 1. Load one DriveLM sample
    # --------------------------------------------------

    parser = DriveLMParser(ANNOTATION_PATH)
    samples = parser.parse()

    sample = samples[0]

    loader = MultiViewImageLoader(NUSCENES_ROOT)
    images = loader.load(sample["images"])

    image = images["CAM_FRONT_RIGHT"]
    question = sample["question"]
    gt_answer = sample["answer"]

    print("=" * 60)
    print("DriveLM Sample")
    print("=" * 60)

    print("Camera:")
    print("CAM_FRONT_RIGHT")

    print("\nImage size:")
    print(image.size)

    print("\nQuestion:")
    print(question)

    print("\nGround Truth:")
    print(gt_answer)

    # --------------------------------------------------
    # 2. Load Qwen3-VL
    # --------------------------------------------------

    print("\nLoading Qwen3-VL...")

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
    )

    model.eval()

    # --------------------------------------------------
    # 3. Construct multimodal message
    # --------------------------------------------------

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": question,
                },
            ],
        }
    ]

    # --------------------------------------------------
    # 4. Processor
    # --------------------------------------------------

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    print("\n" + "=" * 60)
    print("Processor Output")
    print("=" * 60)

    for key, value in inputs.items():
        if hasattr(value, "shape"):
            print(f"{key:20s}: {tuple(value.shape)}")
            
    print("image_grid_thw value:", inputs["image_grid_thw"])

    # --------------------------------------------------
    # 5. GPU memory before generation
    # --------------------------------------------------

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3

    print(f"\nGPU allocated: {allocated:.2f} GB")
    print(f"GPU reserved : {reserved:.2f} GB")

    # --------------------------------------------------
    # 6. Generate
    # --------------------------------------------------

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )

    generated_ids = generated_ids[
        :, inputs["input_ids"].shape[1]:
    ]

    prediction = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    # --------------------------------------------------
    # 7. Result
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("Result")
    print("=" * 60)

    print("\nQuestion:")
    print(question)

    print("\nPrediction:")
    print(prediction)

    print("\nGround Truth:")
    print(gt_answer)

if __name__ == "__main__":
    main()