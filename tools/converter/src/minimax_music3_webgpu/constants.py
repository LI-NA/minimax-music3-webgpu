"""Pinned source-model constants."""

MODEL_ID = "MiniMaxAI/MiniMax-Music3"
MODEL_REVISION = "fbdf52fbaaca799592917417eb05f1899f1255ec"
DIFFUSERS_REVISION = "3681e65996b4d2589219720101a6acbfd25073f8"
ACOUSTIC_SOURCE_FILES = (
    "condition_encoder/config.json",
    "condition_encoder/diffusion_pytorch_model.safetensors",
    "rvq_depth_decoder/config.json",
    "rvq_depth_decoder/diffusion_pytorch_model.safetensors",
    "scheduler/scheduler_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
    "vocoder/config.json",
    "vocoder/diffusion_pytorch_model.safetensors",
)
ARTIFACT_FILE_LIMIT = 128 * 1024 * 1024
HIDDEN_SIZE = 4096
VOCAB_SIZE = 200_000
AUDIO_END_TOKEN_ID = 151_670
SEMANTIC_TOKEN_START = 151_675
SEMANTIC_TOKEN_COUNT = 16_384
