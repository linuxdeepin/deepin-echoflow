"""moss-tts-local reference backend — stub.
Shares architecture with moss-tts (Qwen3 backbone + MOSS tokenizer).
Ref-dump uses the moss_tts.py backend with the local transformer variant."""
DEFAULT_STAGES = ["encoder_output"]
def dump(model_dir, audio, stages, **kwargs):
    raise NotImplementedError("moss-tts-local: use moss_tts.py with local model")
