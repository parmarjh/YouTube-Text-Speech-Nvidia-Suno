# Audio Transcription
Audio transcription tool using `Nvidia's` latest model `Parakeet RNNT 1.1B (en)`. This model is jointly developed by [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) and [Suno.ai](https://www.suno.ai/) teams. It is an XXL version of FastConformer Transducer [1] (around 1.1B parameters) model.
See the [model architecture](#model-architecture) section and [NeMo documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#fast-conformer) for complete architecture details.

* [Model](https://huggingface.co/nvidia/parakeet-rnnt-1.1b)
* [Nvidia NeMo](https://github.com/NVIDIA/NeMo)

### Python package installation steps:
Note: if you are not using uv for the python environment management then just remove the `uv` word from the following package installation steps:

```bash
uv pip install gradio
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install "nemo_toolkit[all]"
```

### Running the application
```bash
gradio main.py
```