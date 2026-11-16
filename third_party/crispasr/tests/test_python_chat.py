#!/usr/bin/env python3
"""Integration tests for the CrispASR Python chat binding (crispasr_chat.h).

Gated on CRISPASR_CHAT_TEST_MODEL — a path to a small GGUF chat model
(e.g. gemma-3-1b-it-Q4_K_M.gguf, qwen2.5-0.5b-instruct, smollm2-360m).
When it is unset every case here is skipped, so a checkout without a model
stays green. The same gate the Catch2 suite uses (tests/test-chat-ggml.cpp).

The native library is found the way the binding itself finds it; set
CRISPASR_LIB_PATH (or CRISPASR_LIB) to point at a build tree copy, e.g.

  CRISPASR_LIB_PATH=build-tests/src/libcrispasr.dylib \\
  CRISPASR_CHAT_TEST_MODEL=models/gemma-3-1b-it-Q4_K_M.gguf \\
  pytest tests/test_python_chat.py -v

What the ctypes-specific cases pin, beyond the surface working at all:

  • generation runs with the GIL released, so a Python thread keeps running
    while the model decodes;
  • an exception raised inside a callback is captured and re-raised after the
    native call returns, never unwound through C;
  • should_continue() returns True to KEEP GOING, the same way round as the C
    callback — proved from both sides, since either half alone passes under an
    inverted implementation.

Run:
  python tests/test_python_chat.py
  pytest tests/test_python_chat.py -v
"""

import ctypes
import os
import sys
import threading
import time
import unittest
import weakref

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")

CHAT_MODEL = os.environ.get("CRISPASR_CHAT_TEST_MODEL")

# The binding probes $CRISPASR_LIB_PATH itself; $CRISPASR_LIB is what the
# other Python tests in this tree use, and a build-tests tree is where the
# ctest configuration puts the shared library.
LIB_PATH = os.environ.get("CRISPASR_LIB_PATH") or os.environ.get("CRISPASR_LIB")
if not LIB_PATH:
    for candidate in [
        os.path.join(REPO_ROOT, "build-tests", "src", "libcrispasr.dylib"),
        os.path.join(REPO_ROOT, "build-tests", "src", "libcrispasr.so"),
        os.path.join(REPO_ROOT, "build", "src", "libcrispasr.dylib"),
        os.path.join(REPO_ROOT, "build", "src", "libcrispasr.so"),
    ]:
        if os.path.exists(candidate):
            LIB_PATH = candidate
            break

USER = [{"role": "user", "content": "Count from one to twenty, one word per line."}]

# A prompt whose reply is outside the tokeniser's character vocabulary, so the
# model spells it with byte-fallback tokens: the C side then delivers one chunk
# per BYTE of each character rather than one per character.
MULTIBYTE_USER = [
    {"role": "user", "content": "Reply with exactly this and nothing else: \U0001fabf\U0001facf\U0001fabc"},
]


# A prompt whose greedy reply is fixed and made of short, distinct pieces, so a
# stop substring can be placed inside it and the truncated text pinned exactly.
# The reply this model gives is "1\n2\n3\n4\n5\n6\n7\n8\n".
COUNTING = [
    {"role": "user",
     "content": "Count from 1 to 8. Write only the numbers, one per line, nothing else."},
]

# What COUNTING yields once generation stops on "4" — the text the caller
# receives, with the match itself cut off. The Rust and Go chat suites assert
# this same string for the same prompt, stop list and sampler settings: three
# separate marshallings of one C feature, agreeing byte for byte.
COUNTING_STOPPED_AT_FOUR = "1\n2\n3\n"

# The literal above is one MODEL's greedy reply, not a property of the stop
# feature, and the gate accepts any small chat GGUF — the docstring names three.
# On smollm2-360m-instruct the same prompt answers "1 2 3 4 ..." with spaces, so
# four cases in TestChatStopSequences failed on the separator while every
# behavioural assertion beside them passed: a test failing for a reason
# unrelated to the code under test. The cross-binding oracle is worth keeping
# (Rust, Go, Java and Dart pin the same strings, so four marshallings of one C
# feature are held byte-identical), so the class now CHECKS it is running on the
# model those strings describe and skips if not, rather than reporting a red.
COUNTING_BASELINE_REPLY = "1\n2\n3\n4\n5\n6\n7\n8\n"


def _has_splittable_char(text):
    """True when ``text`` holds a character the streamed path could split — a
    multi-byte one that is not itself a replacement character."""
    return any(len(ch.encode("utf-8")) > 1 and ch != "�" for ch in text)


class CallbackMarker(Exception):
    """Raised from inside a callback, so a test can recognise it again."""


@unittest.skipUnless(LIB_PATH and os.path.exists(LIB_PATH), "libcrispasr not built")
class TestChatSymbols(unittest.TestCase):
    """Symbol reachability — no model, so it runs on a checkout that has none.

    Everything else in this file needs a GGUF; this is what still notices a
    chat entry point disappearing from the library the binding declares.
    """

    SYMBOLS = [
        "crispasr_chat_ai_disclosure_text",
        "crispasr_chat_close",
        "crispasr_chat_count_tokens",
        "crispasr_chat_generate",
        "crispasr_chat_generate_params_default",
        "crispasr_chat_generate_stream",
        "crispasr_chat_memory_estimate",
        "crispasr_chat_n_ctx",
        "crispasr_chat_open",
        "crispasr_chat_open_params_default",
        "crispasr_chat_reset",
        "crispasr_chat_set_abort_callback",
        "crispasr_chat_string_free",
        "crispasr_chat_template_name",
    ]

    def test_every_chat_symbol_is_exported(self):
        lib = ctypes.CDLL(LIB_PATH)
        missing = [name for name in self.SYMBOLS if not hasattr(lib, name)]
        self.assertEqual(missing, [])

    def test_the_binding_declares_them(self):
        from crispasr import _binding
        lib = _binding._chat_lib(LIB_PATH)
        # _chat_lib is where every argtype/restype is declared; a symbol it
        # forgot has no argtypes and would be called with ctypes' defaults.
        for name in self.SYMBOLS:
            self.assertIsNotNone(getattr(lib, name).argtypes, name)

    def test_memory_estimate_returns_a_full_width_size(self):
        """The estimate is a size_t and has to be read as one: at a modern
        model's own trained context the figure runs past 2**32, so a
        narrower restype would silently truncate it into a number that fits.
        """
        from crispasr import ChatSession, _binding
        lib = _binding._chat_lib(LIB_PATH)
        self.assertIs(lib.crispasr_chat_memory_estimate.restype, ctypes.c_size_t)
        self.assertTrue(callable(ChatSession.memory_estimate))


class _FakeChatLib:
    """Stands in for the loaded CDLL, so the callback-lifetime cases below need
    neither a model nor a build.

    It records a WEAK reference to every abort trampoline the binding hands to
    "C" — weak, because a strong one would itself keep the trampoline alive and
    hide the very thing under test — and its generate entry point parks the
    first caller until the test lets it go. That park is the window a real
    generation spends with those function pointers live inside C.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.calls = 0
        self.entered = threading.Event()
        self.release = threading.Event()
        self.registered = []
        self.closed = False

    def crispasr_chat_generate_params_default(self, _params):
        return None

    def crispasr_chat_set_abort_callback(self, _handle, cb, _user):
        # Clearing passes a NULL instance of the callback type; only a real
        # registration means C is holding a pointer.
        if cb:
            self.registered.append(weakref.ref(cb))
        return None

    def crispasr_chat_generate_stream(self, _handle, _msgs, _n, _params,
                                      _token_cb, _user, _err):
        with self.lock:
            self.calls += 1
            first = self.calls == 1
        self.entered.set()
        if first:
            self.release.wait(10)
        return 0

    def crispasr_chat_close(self, _handle):
        self.closed = True


class _ParkingChatLib(_FakeChatLib):
    """A fake that parks inside a non-generate entry point.

    These are the calls that take no part in the one-generation-at-a-time
    lock. ``crispasr_chat_close`` takes no session mutex either — it frees the
    context, the model and the session outright — so a close that waited only
    for the generate pair would free the handle under one of these.
    """

    def _park(self):
        self.entered.set()
        self.release.wait(10)

    def crispasr_chat_count_tokens(self, _handle, _msgs, _n, _err):
        self._park()
        return 7

    def crispasr_chat_reset(self, _handle, _err):
        self._park()
        return 0

    def crispasr_chat_n_ctx(self, _handle):
        self._park()
        return 4096

    def crispasr_chat_template_name(self, _handle):
        self._park()
        return b"gemma"


def _fake_session(lib):
    """A ChatSession over ``lib`` with no model behind it — the state
    __init__ sets, without the open.

    Through _init_state rather than a hand-written copy of it: the locking
    these cases exercise is exactly what would drift if the two lists of
    attributes were maintained separately."""
    from crispasr import ChatSession
    chat = ChatSession.__new__(ChatSession)
    chat._init_state()
    chat._lib = lib
    chat._handle = 1
    return chat


class TestChatCallbackLifetime(unittest.TestCase):
    """The abort trampoline must outlive the native call that registered it.

    ctypes keeps no reference to a CFUNCTYPE object it hands to C, so whatever
    the binding holds it in IS the lifetime. Holding it in an instance
    attribute makes that lifetime "until the next call on this session
    overwrites the slot", which is shorter than the generation still calling
    through it — a use-after-free that no amount of C-side locking can fix,
    because the free happens on the Python side before C is even asked.

    These run without a model: the fake library above supplies the one thing
    that matters, a native call that has not returned yet.
    """

    def _start_first_call(self, lib, chat):
        """Run a generation on another thread and return once it is inside the
        native call, with its abort trampoline registered."""
        failures = []

        def run():
            try:
                chat.generate_stream(USER, lambda _t: None,
                                     should_continue=lambda: True)
            except BaseException as e:  # noqa: BLE001 - reported by the caller
                failures.append(e)

        thread = threading.Thread(target=run)
        thread.start()
        self.assertTrue(lib.entered.wait(10), "the first call never reached C")
        self.assertEqual(len(lib.registered), 1)
        return thread, failures

    def test_a_second_call_does_not_free_the_running_call_s_trampoline(self):
        lib = _FakeChatLib()
        chat = _fake_session(lib)
        thread, failures = self._start_first_call(lib, chat)
        live = lib.registered[0]
        self.assertIsNotNone(live(), "setup: the trampoline was already gone")

        second = []

        def again():
            try:
                chat.generate_stream(USER, lambda _t: None,
                                     should_continue=lambda: True)
                second.append(None)
            except BaseException as e:  # noqa: BLE001 - inspected below
                second.append(e)

        intruder = threading.Thread(target=again)
        intruder.start()
        intruder.join(10)
        # Read it while the first call is still parked in C, which is the only
        # moment the question means anything.
        survived = live() is not None

        lib.release.set()
        thread.join(10)
        intruder.join(10)
        self.assertEqual(failures, [])
        self.assertTrue(
            survived,
            "the second call dropped the last reference to the trampoline the "
            "first call had handed to C, which is still calling through it")

    def test_a_second_concurrent_call_is_refused_not_queued(self):
        lib = _FakeChatLib()
        chat = _fake_session(lib)
        thread, failures = self._start_first_call(lib, chat)

        with self.assertRaises(RuntimeError) as caught:
            chat.generate_stream(USER, lambda _t: None)
        self.assertIn("one call at a time", str(caught.exception))
        # Refused before reaching C, so the entry point saw only the one call.
        self.assertEqual(lib.calls, 1)

        lib.release.set()
        thread.join(10)
        self.assertEqual(failures, [])

    def test_close_waits_for_a_running_call(self):
        lib = _FakeChatLib()
        chat = _fake_session(lib)
        thread, failures = self._start_first_call(lib, chat)
        live = lib.registered[0]

        closer = threading.Thread(target=chat.close)
        closer.start()
        closer.join(0.5)
        still_running = closer.is_alive()
        freed_under_the_call = lib.closed
        trampoline_gone = live() is None

        lib.release.set()
        thread.join(10)
        closer.join(10)

        self.assertEqual(failures, [])
        self.assertTrue(still_running, "close() returned while a call was running")
        self.assertFalse(freed_under_the_call,
                         "crispasr_chat_close ran under a live generation")
        self.assertFalse(trampoline_gone,
                         "close() dropped a trampoline C was still holding")
        self.assertTrue(lib.closed, "close() never ran once the call finished")
        self.assertIsNone(chat._handle)

    def test_close_waits_for_a_call_outside_the_generate_pair(self):
        """count_tokens, reset and the properties hold the handle too.

        They do not take the one-generation-at-a-time lock — on purpose, so a
        read-only accessor never blocks behind a running generation — so
        close has to wait on something they DO take, or it frees the session
        while C is inside one of them.
        """
        cases = (
            ("count_tokens", lambda c: c.count_tokens(USER)),
            ("reset", lambda c: c.reset()),
            ("n_ctx", lambda c: c.n_ctx),
            ("template_name", lambda c: c.template_name),
        )
        for name, run in cases:
            with self.subTest(method=name):
                lib = _ParkingChatLib()
                chat = _fake_session(lib)
                failures = []

                def call():
                    try:
                        run(chat)
                    except BaseException as e:  # noqa: BLE001 - asserted below
                        failures.append(e)

                thread = threading.Thread(target=call)
                thread.start()
                self.assertTrue(lib.entered.wait(10),
                                f"{name} never reached C")

                closer = threading.Thread(target=chat.close)
                closer.start()
                closer.join(0.5)
                still_running = closer.is_alive()
                freed_under_the_call = lib.closed

                # A call arriving once close has started is refused rather
                # than queued behind it: the handle is already retired.
                with self.assertRaises(RuntimeError) as caught:
                    chat.count_tokens(USER)
                self.assertIn("closed", str(caught.exception))

                lib.release.set()
                thread.join(10)
                closer.join(10)

                self.assertEqual(failures, [])
                self.assertTrue(still_running,
                                f"close() returned while {name} was running")
                self.assertFalse(freed_under_the_call,
                                 f"crispasr_chat_close ran under a live {name}")
                self.assertTrue(lib.closed,
                                "close() never ran once the call finished")

    def test_a_second_call_is_allowed_once_the_first_has_returned(self):
        """The refusal is about overlap, not a session that gets used twice."""
        lib = _FakeChatLib()
        chat = _fake_session(lib)
        lib.release.set()
        chat.generate_stream(USER, lambda _t: None, should_continue=lambda: True)
        chat.generate_stream(USER, lambda _t: None, should_continue=lambda: True)
        self.assertEqual(lib.calls, 2)


@unittest.skipUnless(CHAT_MODEL, "CRISPASR_CHAT_TEST_MODEL not set")
class TestChatSession(unittest.TestCase):
    """One session for the whole class — loading a GGUF per case is slow.

    Every case resets first, so each generation prefills from scratch and none
    of them depends on what the previous one left in the KV cache.
    """

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(CHAT_MODEL):
            raise unittest.SkipTest(f"chat model not found: {CHAT_MODEL}")
        from crispasr import ChatSession
        cls.chat = ChatSession(CHAT_MODEL, lib_path=LIB_PATH, n_ctx=1024)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "chat"):
            cls.chat.close()

    def setUp(self):
        self.chat.reset()

    # -- surface ----------------------------------------------------------

    def test_open_reports_context_size_and_template(self):
        self.assertGreater(self.chat.n_ctx, 0)
        self.assertNotEqual(self.chat.template_name, "")

    def test_generate_returns_text(self):
        out = self.chat.generate(USER, max_tokens=32, temperature=0.0)
        self.assertIsInstance(out, str)
        self.assertNotEqual(out.strip(), "")

    def test_stream_chunks_concatenate_to_the_one_shot_output(self):
        params = dict(max_tokens=32, temperature=0.0, seed=1234)
        one_shot = self.chat.generate(USER, **params)

        self.chat.reset()
        chunks = []
        self.chat.generate_stream(USER, chunks.append, **params)
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks), one_shot)

    def test_stream_rebuilds_characters_split_across_chunks(self):
        params = dict(max_tokens=32, temperature=0.0, seed=1234)
        one_shot = self.chat.generate(MULTIBYTE_USER, **params)
        if not _has_splittable_char(one_shot):
            self.skipTest(f"this model answered {one_shot!r}, with nothing to split")

        self.chat.reset()
        chunks = []
        self.chat.generate_stream(MULTIBYTE_USER, chunks.append, **params)
        self.assertEqual(
            "".join(chunks).encode("utf-8"), one_shot.encode("utf-8"),
            "a character split across chunks must survive")

    def test_stream_delivers_a_character_the_token_budget_cut_in_half(self):
        # Two tokens is under one byte-fallback character's worth for a model
        # that has to spell the reply out byte by byte, so the reply stops
        # part-way through a character and those bytes can never be completed.
        params = dict(max_tokens=2, temperature=0.0, seed=1234)
        one_shot = self.chat.generate(MULTIBYTE_USER, **params)
        if "�" not in one_shot:
            self.skipTest(f"this model stopped on a character boundary, at {one_shot!r}")

        self.chat.reset()
        chunks = []
        self.chat.generate_stream(MULTIBYTE_USER, chunks.append, **params)
        self.assertEqual(
            "".join(chunks).encode("utf-8"), one_shot.encode("utf-8"),
            "the half character must not be dropped")

    def test_count_tokens_is_positive_and_monotone(self):
        short = self.chat.count_tokens([{"role": "user", "content": "Hi."}])
        self.assertGreater(short, 0)

        longer = self.chat.count_tokens([
            {"role": "user", "content": "Hi."},
            {"role": "assistant", "content": "Hello! How can I help you today?"},
            {"role": "user", "content": "Tell me about the sea."},
        ])
        self.assertGreater(longer, short)
        # It is the prompt a fresh session prefills, so it has to fit the window.
        self.assertLess(longer, self.chat.n_ctx)

    def test_count_tokens_accepts_the_message_dataclass(self):
        from crispasr import ChatMessage
        as_dict = self.chat.count_tokens([{"role": "user", "content": "Hi."}])
        as_obj = self.chat.count_tokens([ChatMessage(role="user", content="Hi.")])
        self.assertEqual(as_dict, as_obj)

    def test_ai_disclosure_text_is_available(self):
        text = self.chat.ai_disclosure_text(lib_path=LIB_PATH)
        self.assertNotEqual(text.strip(), "")

    # -- abort ------------------------------------------------------------

    def test_abort_stops_the_stream_early_and_the_session_is_reusable(self):
        from crispasr import ChatAborted

        chunks = []

        def should_continue():
            return len(chunks) < 3

        with self.assertRaises(ChatAborted) as caught:
            self.chat.generate_stream(USER, chunks.append, should_continue=should_continue,
                                      max_tokens=128, temperature=0.0)
        # A cancel is not a decode fault, and it is still a RuntimeError so
        # existing `except RuntimeError` callers keep working.
        self.assertIsInstance(caught.exception, RuntimeError)
        self.assertGreater(len(chunks), 0)
        self.assertLess(len(chunks), 128)

        # An abort flushes the session back to its just-opened state, so this
        # generation runs with no reset() in between.
        again = self.chat.generate(USER, max_tokens=16, temperature=0.0)
        self.assertNotEqual(again.strip(), "")

    def test_abort_polarity_is_true_means_keep_going(self):
        """Two-sided: either half alone passes under an inverted binding."""
        from crispasr import ChatAborted

        never = []
        self.chat.generate_stream(USER, never.append, should_continue=lambda: True,
                                  max_tokens=24, temperature=0.0)
        self.assertGreater(len(never), 1)

        self.chat.reset()
        always = []
        with self.assertRaises(ChatAborted):
            self.chat.generate_stream(USER, always.append, should_continue=lambda: False,
                                      max_tokens=24, temperature=0.0)
        self.assertLess(len(always), len(never))

    def test_abort_predicate_result_matches_the_one_shot_path(self):
        from crispasr import ChatAborted
        with self.assertRaises(ChatAborted):
            self.chat.generate(USER, should_continue=lambda: False,
                               max_tokens=24, temperature=0.0)

    # -- argument validation ----------------------------------------------

    def test_an_interior_nul_is_rejected_rather_than_truncated(self):
        """C reads a NUL as the end of a string, so passing one through would
        silently drop the rest of a message, stop sequence or model path.
        Rust's CString::new rejects the same input; this binding raises
        ValueError naming the field it came from."""
        with self.assertRaises(ValueError) as caught:
            self.chat.count_tokens([{"role": "user", "content": "before\x00after"}])
        self.assertIn("messages[0].content", str(caught.exception))

        with self.assertRaises(ValueError) as caught:
            self.chat.count_tokens([{"role": "user", "content": "fine"},
                                    {"role": "us\x00er", "content": "hi"}])
        self.assertIn("messages[1].role", str(caught.exception))

        with self.assertRaises(ValueError) as caught:
            self.chat.generate(USER, max_tokens=8, temperature=0.0, stop=["ok", "sto\x00p"])
        self.assertIn("stop[1]", str(caught.exception))

        # The model path too, on both entry points that take one: truncating
        # it at the NUL would open a different file from the one named — or,
        # worse for a guard, estimate one.
        from crispasr import ChatSession
        bad_path = CHAT_MODEL + "\x00.gguf"
        with self.assertRaises(ValueError) as caught:
            ChatSession(bad_path, lib_path=LIB_PATH)
        self.assertIn("model_path", str(caught.exception))

        with self.assertRaises(ValueError) as caught:
            ChatSession.memory_estimate(bad_path, lib_path=LIB_PATH)
        self.assertIn("model_path", str(caught.exception))

    # -- callback lifetime ------------------------------------------------

    def test_callback_exception_surfaces_after_the_native_call(self):
        """Nothing unwinds through C, and the failure cancels the run.

        Two things are checked, because "the exception reached me" alone does
        not distinguish the two paths. If the exception had escaped the ctypes
        callback into C, Python would have reported it through
        sys.unraisablehook and the call would have returned normally — so an
        empty unraisable log plus a raise from the method is the capture path.
        And once the token callback has raised, the abort predicate is not
        consulted again: the trampoline answers "stop" on its own, which is
        what turns the failure into a cancellation. The Go and Rust bindings
        behave the same way.
        """
        unraisable = []
        previous_hook = sys.unraisablehook
        sys.unraisablehook = unraisable.append

        abort_calls = []
        calls_when_raised = []
        chunks = []

        def should_continue():
            abort_calls.append(1)
            return True

        def on_token(chunk):
            chunks.append(chunk)
            if len(chunks) == 1:
                calls_when_raised.append(len(abort_calls))
                raise CallbackMarker("from the token callback")

        try:
            with self.assertRaises(CallbackMarker):
                self.chat.generate_stream(USER, on_token, should_continue=should_continue,
                                          max_tokens=64, temperature=0.0)
        finally:
            sys.unraisablehook = previous_hook

        self.assertEqual(unraisable, [])
        self.assertEqual(len(chunks), 1, "chunks after the raise must be dropped")
        # Positive control first: without it the equality below would also hold
        # for a predicate that was never registered.
        self.assertGreater(calls_when_raised[0], 0,
                           "the predicate was never consulted, so the count proves nothing")
        self.assertEqual(len(abort_calls), calls_when_raised[0],
                         "the predicate must not be consulted after the token callback raised")

        # The session survives it: the failure was delivered as a cancel.
        self.assertNotEqual(self.chat.generate(USER, max_tokens=16, temperature=0.0).strip(), "")

    def test_abort_predicate_exception_surfaces_and_is_not_called_again(self):
        calls = []

        def should_continue():
            calls.append(1)
            raise CallbackMarker("from the abort predicate")

        with self.assertRaises(CallbackMarker):
            self.chat.generate(USER, should_continue=should_continue,
                               max_tokens=64, temperature=0.0)
        self.assertEqual(len(calls), 1)

    def test_generate_releases_the_gil(self):
        """ctypes drops the GIL around a CDLL call; a plain thread proves it.

        The one-shot path fires no Python callback at all, so if the GIL were
        held for the duration of the native call the counter thread could not
        advance a single tick while the model decodes.
        """
        stop = threading.Event()
        ticks = [0]

        def spin():
            while not stop.is_set():
                ticks[0] += 1
                time.sleep(0.001)

        worker = threading.Thread(target=spin, daemon=True)
        worker.start()
        try:
            time.sleep(0.05)          # let the thread reach the loop
            before = ticks[0]
            started = time.monotonic()
            self.chat.generate(USER, max_tokens=48, temperature=0.0)
            elapsed = time.monotonic() - started
            during = ticks[0] - before
        finally:
            stop.set()
            worker.join(timeout=2)

        self.assertGreater(elapsed, 0.05, "generation too quick to say anything about the GIL")
        self.assertGreater(during, 5, "the counter thread never ran during generation")


@unittest.skipUnless(CHAT_MODEL, "CRISPASR_CHAT_TEST_MODEL not set")
class TestChatSessionLifecycle(unittest.TestCase):
    """Opening and dropping sessions, away from the shared-session class."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(CHAT_MODEL):
            raise unittest.SkipTest(f"chat model not found: {CHAT_MODEL}")

    def test_context_manager_closes_and_a_second_session_opens(self):
        from crispasr import ChatSession
        with ChatSession(CHAT_MODEL, lib_path=LIB_PATH, n_ctx=512) as chat:
            first = chat.generate(USER, max_tokens=8, temperature=0.0)
        self.assertNotEqual(first.strip(), "")

        with ChatSession(CHAT_MODEL, lib_path=LIB_PATH, n_ctx=512) as chat:
            self.assertGreater(chat.n_ctx, 0)

    def test_memory_estimate_covers_the_weights_and_scales_with_context(self):
        from crispasr import ChatSession
        file_size = os.path.getsize(CHAT_MODEL)
        self.assertGreater(file_size, 0)

        def at(n_ctx=None):
            return ChatSession.memory_estimate(
                CHAT_MODEL, lib_path=LIB_PATH, n_ctx=n_ctx)

        # n_ctx left unset: the model's own trained context sizes the KV term.
        self.assertGreater(at(), file_size)

        at_1k, at_2k, at_4k = at(1024), at(2048), at(4096)
        self.assertGreater(at_1k, file_size)
        self.assertGreater(at_2k, at_1k)
        self.assertGreater(at_4k, at_2k)

        # The KV term is linear in n_ctx, so doubling the context doubles the
        # amount by which the estimate grows. A load path that returned before
        # reading the context / layer / embedding metadata would leave every
        # difference at zero and still report success.
        self.assertEqual(at_4k - at_2k, 2 * (at_2k - at_1k),
                         f"KV term not linear in n_ctx: {at_1k} / {at_2k} / {at_4k}")

        # Everything outside the KV term is context-independent, so back it
        # out and the remainder still has to cover the weights on disk.
        self.assertGreater(at_1k - (at_2k - at_1k), file_size)

    def test_memory_estimate_raises_for_a_model_it_cannot_read(self):
        """The C side signals failure by returning 0 with err filled. That has
        to surface as an exception, not as an estimate of nothing."""
        from crispasr import ChatSession
        with self.assertRaises(RuntimeError):
            ChatSession.memory_estimate(
                "/nonexistent/crispasr-memory-estimate.gguf", lib_path=LIB_PATH)

    def test_a_chat_template_with_an_interior_nul_is_rejected(self):
        """Rejected before the model is touched, so the bad template cannot
        reach C truncated."""
        from crispasr import ChatSession
        with self.assertRaises(ValueError) as caught:
            ChatSession(CHAT_MODEL, lib_path=LIB_PATH, n_ctx=512,
                        chat_template="{{ bos_token }}\x00{{ messages }}")
        self.assertIn("chat_template", str(caught.exception))

    def test_open_failure_raises(self):
        from crispasr import ChatSession
        with self.assertRaises(RuntimeError):
            ChatSession(os.path.join(REPO_ROOT, "no-such-model.gguf"), lib_path=LIB_PATH)

    def test_a_real_generation_refuses_a_second_thread(self):
        """The same refusal as the fake-library case, against a real model.

        Deterministic, not a race: the running generation's own predicate
        parks until the second thread has had its turn, so the first call
        cannot finish before the second one tries.
        """
        from crispasr import ChatSession
        inside = threading.Event()
        attempted = threading.Event()
        outcome = []

        with ChatSession(CHAT_MODEL, lib_path=LIB_PATH, n_ctx=512) as chat:
            def other():
                if not inside.wait(30):
                    return
                try:
                    chat.generate(USER, max_tokens=8, temperature=0.0)
                    outcome.append(None)
                except BaseException as e:  # noqa: BLE001 - inspected below
                    outcome.append(e)
                finally:
                    attempted.set()

            # The predicate runs before every token; park on the first call
            # only, so a binding that queues the second thread instead of
            # refusing it fails the assertions below rather than parking once
            # per token.
            parked = []

            def keep_going():
                if not parked:
                    parked.append(True)
                    inside.set()
                    attempted.wait(30)
                return True

            thread = threading.Thread(target=other)
            thread.start()
            try:
                first = chat.generate(USER, max_tokens=16, temperature=0.0,
                                      should_continue=keep_going)
            finally:
                attempted.set()
                thread.join(30)

            self.assertNotEqual(first.strip(), "")
            self.assertEqual(len(outcome), 1)
            self.assertIsInstance(outcome[0], RuntimeError)
            self.assertIn("one call at a time", str(outcome[0]))
            # The session is unharmed by the refusal.
            self.assertNotEqual(chat.generate(USER, max_tokens=8,
                                              temperature=0.0).strip(), "")


@unittest.skipUnless(CHAT_MODEL, "CRISPASR_CHAT_TEST_MODEL not set")
class TestChatStopSequences(unittest.TestCase):
    """The ``stop`` array and ``prefill_only``.

    A session of its own, with the context and batch sizes pinned to the ones
    the Rust and Go suites use, so the truncated text below is comparable
    across all three bindings rather than merely reproducible in this one.
    """

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(CHAT_MODEL):
            raise unittest.SkipTest(f"chat model not found: {CHAT_MODEL}")
        from crispasr import ChatSession
        cls.chat = ChatSession(CHAT_MODEL, lib_path=LIB_PATH,
                               n_ctx=2048, n_batch=256, n_ubatch=256)


    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "chat"):
            cls.chat.close()

    def setUp(self):
        self.chat.reset()

    def _require_pinned_baseline(self):
        """Skip unless the gated model is the one the literals below describe.

        Called only by the cases that assert an exact string, so the
        model-independent ones (empty stop list, prefill-only) still run on any
        gate model.
        """
        self.chat.reset()
        baseline = self.chat.generate(COUNTING, max_tokens=64, temperature=0.0)
        self.chat.reset()
        if baseline != COUNTING_BASELINE_REPLY:
            self.skipTest(
                "literal stop-sequence assertions are pinned to the model whose greedy reply is "
                f"{COUNTING_BASELINE_REPLY!r} (e.g. gemma-3-1b-it-Q4_K_M); this model "
                f"replies {baseline!r}. Behaviour is covered model-independently by "
                "tests/test-chat-ggml.cpp.")

    def generate(self, **kwargs):
        self.chat.reset()
        return self.chat.generate(COUNTING, max_tokens=64, temperature=0.0, **kwargs)

    def test_a_stop_sequence_truncates_the_reply_before_the_match(self):
        self._require_pinned_baseline()
        full = self.generate()
        # Without this the case is vacuous: a reply that never reaches the
        # stop string would pass whether or not stop sequences work at all.
        self.assertIn("5", full)

        stopped = self.generate(stop=["5"])
        self.assertNotIn("5", stopped)
        self.assertTrue(full.startswith(stopped), f"{stopped!r} vs {full!r}")
        self.assertEqual(stopped, "1\n2\n3\n4\n")

    def test_several_stop_sequences_stop_at_the_earliest_match(self):
        self._require_pinned_baseline()
        full = self.generate()
        self.assertIn("4", full)
        self.assertIn("7", full)

        # Both orders: "4" is generated before "7", so "4" wins either way —
        # the earliest match in the output decides, not the position in the
        # array. Order-independence is also what says the whole array was
        # marshalled rather than only its first element.
        for stop in (["7", "4"], ["4", "7"]):
            with self.subTest(stop=stop):
                stopped = self.generate(stop=stop)
                self.assertEqual(stopped, COUNTING_STOPPED_AT_FOUR)
                self.assertNotIn("4", stopped)
                self.assertNotIn("7", stopped)

    def test_an_empty_stop_list_is_the_same_as_none(self):
        none = self.generate()
        empty = self.generate(stop=[])
        self.assertEqual(empty, none)
        # The reply is long enough that a stop list WOULD have truncated it,
        # so the equality above is not two empty strings agreeing.
        self.assertIn("4", none)

    def test_prefill_only_suppresses_generation(self):
        self.assertEqual(self.generate(prefill_only=True), "")

        self.chat.reset()
        chunks = []
        self.chat.generate_stream(COUNTING, chunks.append, max_tokens=64,
                                  temperature=0.0, prefill_only=True)
        self.assertEqual(chunks, [])

        # Positive control: the same messages and sampler settings without the
        # flag do produce text, so the two emptinesses above are the flag's
        # doing and not a prompt that generates nothing.
        self.assertNotEqual(self.generate(), "")

    def test_the_stream_delivers_the_chunk_the_one_shot_path_truncates(self):
        self._require_pinned_baseline()
        params = dict(max_tokens=64, temperature=0.0, stop=["7", "4"])
        one_shot = self.generate(stop=["7", "4"])

        self.chat.reset()
        chunks = []
        self.chat.generate_stream(COUNTING, chunks.append, **params)

        # The C side hands each piece to the callback before it scans for a
        # stop match, so the streamed text carries the matched piece the
        # one-shot return value has cut off.
        self.assertEqual(one_shot, COUNTING_STOPPED_AT_FOUR)
        self.assertEqual("".join(chunks), "1\n2\n3\n4")


if __name__ == "__main__":
    unittest.main()
