"""m2m100 reference backend — stub.
M2M-100 is a text translation model (not audio). Per-stage diff testing
uses crispasr-diff m2m100 directly."""
DEFAULT_STAGES = ["encoder_output", "decoder_output"]
def dump(model_dir, audio, stages, **kwargs):
    raise NotImplementedError("m2m100: text-only model, use crispasr-diff directly")
