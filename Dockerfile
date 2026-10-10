FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV TOKENIZERS_PARALLELISM=false
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN pip install --upgrade pip && \
    pip install \
        "transformers>=4.57.0,<5.19.0" \
        "accelerate>=1.0,<2" \
        "peft>=0.17,<0.19" \
        "bitsandbytes>=0.45,<0.50" \
        "pillow>=10,<13" \
        "numpy>=1.26,<3" \
        "pytest>=8,<10"

CMD ["/bin/bash"]
