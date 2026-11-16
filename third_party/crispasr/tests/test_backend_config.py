import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def read(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


class TestBackendConfig(unittest.TestCase):
    def test_backend_gpu_helper_exists(self) -> None:
        text = read("examples/cli/crispasr_backend_utils.h")
        self.assertIn("inline bool crispasr_backend_should_use_gpu", text)
        self.assertIn('params.gpu_backend != "cpu"', text)

    def test_wrappers_use_shared_gpu_helper(self) -> None:
        wrappers = [
            "examples/cli/crispasr_backend_canary.cpp",
            "examples/cli/crispasr_backend_cohere.cpp",
            "examples/cli/crispasr_backend_firered_asr.cpp",
            "examples/cli/crispasr_backend_glm_asr.cpp",
            "examples/cli/crispasr_backend_granite.cpp",
            "examples/cli/crispasr_backend_kyutai_stt.cpp",
            "examples/cli/crispasr_backend_omniasr.cpp",
            "examples/cli/crispasr_backend_parakeet.cpp",
            "examples/cli/crispasr_backend_qwen3.cpp",
            "examples/cli/crispasr_backend_voxtral.cpp",
            "examples/cli/crispasr_backend_voxtral4b.cpp",
        ]
        for relpath in wrappers:
            text = read(relpath)
            self.assertIn('#include "crispasr_backend_utils.h"', text, relpath)
            self.assertIn("crispasr_backend_should_use_gpu(", text, relpath)
            self.assertNotIn('use_gpu && p.gpu_backend != "cpu"', text, relpath)
            self.assertNotIn('use_gpu && params.gpu_backend != "cpu"', text, relpath)

    def test_glm_encoder_frame_helper_is_declared_and_used(self) -> None:
        header = read("src/glm_asr.h")
        impl = read("src/glm_asr.cpp")

        self.assertIn("int glm_asr_encoder_frames_from_mel_frames(int T_mel);", header)
        self.assertIn("int glm_asr_encoder_frames_from_mel_frames(int T_mel)", impl)
        self.assertIn("return (T_mel + 1) / 2;", impl)
        self.assertEqual(impl.count("glm_asr_encoder_frames_from_mel_frames(T_mel)"), 2)
        self.assertIn("if (T_mel != T_target) {", impl)
        self.assertIn("const int T_valid = std::min(T_mel, T_target);", impl)
        self.assertIn("const int T_pack = T_proj * 4;", impl)

    def test_cli_only_loads_all_backends_when_gpu_is_enabled(self) -> None:
        """`ggml_backend_load_all()` must stay behind BOTH gates.

        Asserted on structure rather than on one exact source line: this test
        used to pin the literal `if (params.use_gpu && params.gpu_backend !=
        "cpu") {` and went red the moment fd3c0e5e split it into a nested pair
        — same behaviour, different formatting, red nightly. What matters is
        that the call is reached only when the GPU is on AND the chosen backend
        is not "cpu"; how the two conditions are spelled is not the contract.
        """
        text = read("examples/cli/cli.cpp")
        start = text.find("if (params.use_gpu)")
        if start < 0:
            start = text.find("if (params.use_gpu &&")
        self.assertGreater(start, 0, "no `if (params.use_gpu)` guard in cli.cpp")
        window = text[start:start + 400]
        self.assertIn('params.gpu_backend != "cpu"', window)
        self.assertIn("ggml_backend_load_all();", window)
        # ...and nowhere else: an unguarded call would defeat the whole point.
        self.assertEqual(text.count("ggml_backend_load_all();"), 1)

    def test_omniasr_keeps_explicit_cpu_fallback_backend(self) -> None:
        """omniasr keeps a CPU backend distinct from the main one, and schedules it.

        Asserted on structure, not on the literal init call — the same lesson the
        sibling test above records. This one pinned
        `ctx->backend_cpu = ggml_backend_cpu_init();` and went red the moment
        f7464aeb (#355) routed every CPU-backend acquisition through
        `core_cpu_backend::init()` so GGML_BACKEND_DL builds can dlopen it. The
        behaviour was unchanged; the nightly was red for five nights over a
        rename.

        The contract is: a CPU fallback exists, it is only used/freed when it is
        genuinely a different object from the main backend, and it is handed to
        the scheduler. How the CPU backend is spelled, and what graph size the
        sched gets, are not the contract.
        """
        text = read("src/omniasr.cpp")
        self.assertRegex(
            text,
            r"ctx->backend_cpu\s*=\s*(?:core_cpu_backend::init|ggml_backend_cpu_init)\s*\(",
            "omniasr must initialise an explicit CPU fallback backend",
        )
        self.assertIn("if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)", text)
        self.assertRegex(
            text,
            r"ggml_backend_sched_new\(\s*backends\s*,\s*nullptr\s*,\s*n_backends\s*,",
            "the CPU fallback must be handed to ggml_backend_sched_new",
        )

    def test_backend_default_params_enable_gpu(self) -> None:
        expected = {
            "src/canary.cpp": "p.use_gpu = true;",
            "src/cohere.cpp": ".use_gpu = true",
            "src/firered_asr.cpp": "/*use_gpu=*/true",
            "src/glm_asr.cpp": "/*use_gpu=*/true",
            "src/granite_speech.cpp": "/*use_gpu=*/true",
            "src/kyutai_stt.cpp": "/*use_gpu=*/true",
            "src/omniasr.cpp": "p.use_gpu = true;",
            "src/parakeet.cpp": "p.use_gpu = true;",
            "src/qwen3_asr.cpp": "p.use_gpu = true;",
            "src/voxtral.cpp": "p.use_gpu = true;",
            "src/voxtral4b.cpp": "/*use_gpu=*/true",
        }
        for relpath, needle in expected.items():
            self.assertIn(needle, read(relpath), relpath)


if __name__ == "__main__":
    unittest.main(verbosity=2)
