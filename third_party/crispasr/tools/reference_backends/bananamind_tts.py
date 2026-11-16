"""bananamind-tts reference backend — stub.
BananaMind TTS uses a Tacotron2-style encoder + WaveRNN decoder.
Per-stage diff testing uses crispasr-diff bananamind-tts directly."""
DEFAULT_STAGES = ["encoder_output"]
def dump(model_dir, audio, stages, **kwargs):
    raise NotImplementedError("bananamind-tts: use crispasr-diff directly")
