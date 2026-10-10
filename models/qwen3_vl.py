import time
import torch

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
)


class Qwen3VLPredictor:
    CAMERA_ORDER = [
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    def __init__(
        self,
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        max_new_tokens=64,
        visual_tokens_per_image=256,
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.visual_tokens_per_image = visual_tokens_per_image

        print(f"Loading model: {model_id}")

        self.processor = AutoProcessor.from_pretrained(
            model_id
        )

        # --------------------------------------------------
        # Visual token budget
        #
        # Qwen3-VL:
        # approximately 32 x 32 input pixels -> 1 visual token
        #
        # 256 tokens/image:
        # 256 * 32 * 32 = 262144 pixels
        # --------------------------------------------------

        max_pixels = visual_tokens_per_image * 32 * 32

        # Avoid excessively shrinking unusual small images.
        min_pixels = 64 * 32 * 32

        self.processor.image_processor.size = {
            "longest_edge": max_pixels,
            "shortest_edge": min_pixels,
        }

        print(
            f"Visual budget: ~{visual_tokens_per_image} "
            f"tokens / image"
        )

        self.model = (
            AutoModelForImageTextToText
            .from_pretrained(
                model_id,
                dtype=torch.float16,
                device_map="cuda",
            )
        )

        self.model.eval()

        print("Model loaded.")

    def _generate(self, messages):
        """
        Shared generation function.
        """

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.model.device)

        # ---------------------------------------------
        # Inspect visual grids
        # ---------------------------------------------

        image_grid_thw = None
        visual_tokens = None

        if "image_grid_thw" in inputs:
            image_grid_thw = (
                inputs["image_grid_thw"]
                .detach()
                .cpu()
            )

            # Qwen3-VL uses 2x2 spatial merge.
            # T * H * W / 4
            visual_tokens = [
                int(grid.prod().item() // 4)
                for grid in image_grid_thw
            ]

        # ---------------------------------------------
        # Profile generation
        # ---------------------------------------------

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        torch.cuda.synchronize()

        latency = time.perf_counter() - start

        peak_memory = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

        # Remove prompt tokens.
        generated_ids = generated_ids[
            :, inputs["input_ids"].shape[1]:
        ]

        prediction = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        result = {
            "prediction": prediction.strip(),
            "latency": latency,
            "peak_memory_gb": peak_memory,
            "input_tokens": int(
                inputs["input_ids"].shape[1]
            ),
        }

        if image_grid_thw is not None:
            result["image_grid_thw"] = (
                image_grid_thw.tolist()
            )

            result["visual_tokens_per_image"] = (
                visual_tokens
            )

            result["total_visual_tokens"] = sum(
                visual_tokens
            )

        return result

    def predict(self, image, question):
        """
        Single-view inference.
        Kept for later experiments/debugging.
        """

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

        return self._generate(messages)

    def predict_multiview(
        self,
        images,
        question,
    ):
        """
        Six-camera surround-view inference.

        images:
            Dict[str, PIL.Image]
        """

        content = [
            {
                "type": "text",
                "text": (
                    "You are observing the surroundings "
                    "of an autonomous vehicle using six "
                    "camera views. Each image is labeled "
                    "with its camera direction relative "
                    "to the ego vehicle.\n\n"
                ),
            }
        ]

        # --------------------------------------------------
        # Add six labeled camera views
        # --------------------------------------------------

        for camera in self.CAMERA_ORDER:

            content.append(
                {
                    "type": "text",
                    "text": f"{camera}:",
                }
            )

            content.append(
                {
                    "type": "image",
                    "image": images[camera],
                }
            )

        # --------------------------------------------------
        # Question
        # --------------------------------------------------

        content.append(
            {
                "type": "text",
                "text": (
                    "\nThe camera labels only indicate where each image was captured "
                    "relative to the ego vehicle. "
                    "Do not answer with camera names. "
                    "Answer the question itself based on the visible physical objects. "
                    "If the question asks what objects are present, name the objects. "
                    "If the question asks for an object's status, state its status. "
                    "Answer in one concise sentence.\n\n"
                    f"Question: {question}"
                ),
            }
        )

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        return self._generate(messages)