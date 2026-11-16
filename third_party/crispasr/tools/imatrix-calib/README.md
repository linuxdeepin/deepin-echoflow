# imatrix calibration set

The audio used to calibrate importance matrices (imatrix) for GGUF ASR
quantisation lives as a dataset on the Hub (not in git — it's audio):

**[`cstr/crispasr-imatrix-calib`](https://huggingface.co/datasets/cstr/crispasr-imatrix-calib)** — CC0, 24 EN + 24 DE clips.

## Provenance / how to rebuild

Clips are the first N mp3s of the `dev` split of **Common Voice 17.0** (CC0),
via the `fsicoli/common_voice_17_0` mirror. The Hub `datasets` loader no longer
runs the dataset script, so pull the raw tar shard and extract directly:

```python
from huggingface_hub import hf_hub_download
import tarfile, os
for lang in ["en", "de"]:
    p = hf_hub_download("fsicoli/common_voice_17_0",
                        f"audio/{lang}/dev/{lang}_dev_0.tar", repo_type="dataset")
    os.makedirs(f"cv/{lang}", exist_ok=True)
    with tarfile.open(p) as t:
        n = 0
        for m in t:
            if not m.name.endswith(".mp3"):
                continue
            open(f"cv/{lang}/{lang}_{n:03d}.mp3", "wb").write(t.extractfile(m).read())
            n += 1
            if n >= 24:
                break
```

crispasr decodes mp3 natively (miniaudio), so feed the clips straight in — see
[`../imatrix_ab.py`](imatrix_ab.py) and [`../../docs/quantize.md`](../../docs/quantize.md).

**Scale up for production:** more clips, and the languages/domains you actually
target — an English-only set *regressed* the imatrix in the A/B; EN+DE won.
