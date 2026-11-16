// chat.dart — Dart binding for the crispasr_chat_* C ABI (text → text LLM).
//
// A session is an opaque `Pointer<Void>` wrapped in a Dart class. Release it
// with `close()`; a `dart:ffi` `NativeFinalizer` holding the session's own
// `crispasr_chat_close` is attached as a backstop for a session whose owner is
// garbage-collected without one. The class implements `Finalizable`, which is
// what keeps a session alive for the whole of any method that hands its handle
// to C.
//
// Generation is one-shot: `generate(messages)` runs the whole reply and
// returns it, optionally under an abort predicate. `crispasr_chat_count_tokens`
// is bound as `countTokens`. `crispasr_chat_generate_stream` is not bound —
// see the note above `close()` for why and what the alternative is.
//
// The header lives in include/crispasr_chat.h and is the authority for every
// shape in this file.

import 'dart:convert';
import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';

import 'crispasr.dart' show CrispASR;

// ---------------------------------------------------------------------------
// FFI shapes — kept private to this file so consumers stay on the
// high-level Dart classes below.
// ---------------------------------------------------------------------------

final class _ChatOpenParams extends Struct {
  @Int32()
  external int nThreads;
  @Int32()
  external int nThreadsBatch;
  @Int32()
  external int nCtx;
  @Int32()
  external int nBatch;
  @Int32()
  external int nUbatch;
  @Int32()
  external int nGpuLayers;
  @Bool()
  external bool useMmap;
  @Bool()
  external bool useMlock;
  @Bool()
  external bool embeddings;
  external Pointer<Utf8> chatTemplate;
}

final class _ChatGenerateParams extends Struct {
  @Int32()
  external int maxTokens;
  @Float()
  external double temperature;
  @Int32()
  external int topK;
  @Float()
  external double topP;
  @Float()
  external double minP;
  @Float()
  external double repeatPenalty;
  @Int32()
  external int repeatLastN;
  @Uint32()
  external int seed;
  external Pointer<Pointer<Utf8>> stop;
  @Size()
  external int nStop;
  @Bool()
  external bool prefillOnly;
}

final class _ChatMessage extends Struct {
  external Pointer<Utf8> role;
  external Pointer<Utf8> content;
}

// Error struct: int32 code + 256-byte message buffer. Total 260 bytes.
// `Array<Int8>` was stabilised in Dart 3.1 (see pubspec.yaml SDK
// constraint); pre-3.1 hosts can't bind this ABI.
final class _ChatError extends Struct {
  @Int32()
  external int code;
  @Array.multi([256])
  external Array<Int8> message;
}

typedef _AbortCallbackNative = Bool Function(Pointer<Void>);

typedef _SetAbortNative = Void Function(Pointer<Void>,
    Pointer<NativeFunction<_AbortCallbackNative>>, Pointer<Void>);
typedef _SetAbortDart = void Function(Pointer<Void>,
    Pointer<NativeFunction<_AbortCallbackNative>>, Pointer<Void>);

/// The one chat error code with a stable meaning: a registered abort
/// predicate stopped the generation.
///
/// Every other non-zero `code` on a [ChatException] is a diagnostic aid, not
/// a contract — read [ChatException.message], do not switch on the number.
/// Callers that run their own cancellation need this one value to tell a
/// cancel apart from a fault; [ChatException.isAborted] is the shorthand.
const int chatErrAborted = 40;

/// Parameters describing how a chat session is opened on top of a GGUF.
class ChatOpenParams {
  /// Generation threads. Defaults to physical-cores cap via the C ABI.
  final int? nThreads;

  /// Batch / prefill threads. Defaults to `nThreads`.
  final int? nThreadsBatch;

  /// Context window in tokens. `null` = model default.
  final int? nCtx;

  /// Logical batch size.
  final int? nBatch;

  /// Physical micro-batch.
  final int? nUbatch;

  /// `-1` = all layers on GPU (default), `0` = CPU only.
  final int? nGpuLayers;
  final bool useMmap;
  final bool useMlock;

  /// Override the chat template baked into the GGUF.
  /// `null` → read `tokenizer.chat_template`, falling back to `chatml`.
  ///
  /// Rejected with an [ArgumentError] if it holds an interior NUL: C reads
  /// it as a null-terminated string, so the rest would be dropped silently.
  final String? chatTemplate;

  const ChatOpenParams({
    this.nThreads,
    this.nThreadsBatch,
    this.nCtx,
    this.nBatch,
    this.nUbatch,
    this.nGpuLayers,
    this.useMmap = true,
    this.useMlock = false,
    this.chatTemplate,
  });
}

/// Per-call sampling configuration.
///
/// Each default below is the value `crispasr_chat_generate_params_default`
/// writes for that field. The two are held together by a test that reads the
/// loaded library's defaults through [ChatGenerateParams.abiDefaults] and
/// compares them field for field against a default-constructed object, so a
/// C-side edit fails here rather than silently leaving this file behind.
class ChatGenerateParams {
  /// Hard cap on tokens generated. `0` does **not** mean "generate nothing":
  /// the C side treats a non-positive `max_tokens` as its own default of 256.
  /// Use [prefillOnly] to prefill without generating.
  final int maxTokens;

  /// `0.0` = greedy.
  final double temperature;

  /// `0` = disabled.
  final int topK;

  /// `1.0` = disabled.
  final double topP;

  /// `0.0` = disabled.
  final double minP;

  /// `1.0` = disabled.
  final double repeatPenalty;

  /// `-1` = context size, `0` = disabled.
  final int repeatLastN;

  /// `0xFFFFFFFF` = random.
  final int seed;

  /// Prefill the system / user portion but suppress assistant generation —
  /// useful for measuring prompt cost. The reply comes back empty.
  final bool prefillOnly;

  /// Stop substrings — generation halts (output truncated before the match)
  /// the first time any of these appears in the accumulated decode. The
  /// earliest match *in the output* wins, not the earliest entry in this
  /// list. An empty list is the same as passing none.
  ///
  /// An entry holding an interior NUL is rejected with an [ArgumentError].
  final List<String> stop;

  const ChatGenerateParams({
    this.maxTokens = 256,
    this.temperature = 0.8,
    this.topK = 40,
    this.topP = 0.95,
    this.minP = 0.05,
    this.repeatPenalty = 1.1,
    this.repeatLastN = 64,
    this.seed = 0,
    this.prefillOnly = false,
    this.stop = const [],
  });

  /// The generate defaults the loaded libcrispasr reports, read from
  /// `crispasr_chat_generate_params_default`.
  ///
  /// Nothing in this file marshals these — the constructor's own defaults are
  /// what a caller gets. This exists so the two can be compared: a test
  /// checks it field for field against `const ChatGenerateParams()`, which is
  /// what turns a C-side default edit into a failure here instead of a silent
  /// divergence. [stop] comes back empty: the ABI default is "no stop
  /// sequences".
  static ChatGenerateParams abiDefaults(
      {DynamicLibrary? lib, String? libPath}) {
    lib ??= DynamicLibrary.open(libPath ?? CrispASR.defaultLibName());
    final ptr = calloc<_ChatGenerateParams>();
    try {
      _generateParamsDefault(lib)(ptr);
      final r = ptr.ref;
      return ChatGenerateParams(
        maxTokens: r.maxTokens,
        temperature: r.temperature,
        topK: r.topK,
        topP: r.topP,
        minP: r.minP,
        repeatPenalty: r.repeatPenalty,
        repeatLastN: r.repeatLastN,
        seed: r.seed,
        prefillOnly: r.prefillOnly,
      );
    } finally {
      calloc.free(ptr);
    }
  }
}

/// One message in a chat conversation. Same shape as the OpenAI
/// chat-completions schema; the chat-template layer translates
/// `role` into whatever the model's GGUF template expects.
///
/// Neither field may hold an interior NUL — see [ChatOpenParams.chatTemplate].
class ChatMessage {
  final String role;
  final String content;
  const ChatMessage({required this.role, required this.content});

  factory ChatMessage.system(String content) =>
      ChatMessage(role: 'system', content: content);
  factory ChatMessage.user(String content) =>
      ChatMessage(role: 'user', content: content);
  factory ChatMessage.assistant(String c) =>
      ChatMessage(role: 'assistant', content: c);
}

/// A chat call failed. [code] is the C side's `crispasr_chat_error.code`,
/// falling back to the call's own return value when the struct was left at
/// zero.
///
/// Only [chatErrAborted] is a contract. Test [isAborted], or catch
/// [ChatAborted], to tell a cancellation from a fault; for anything else read
/// [message] rather than switching on [code].
class ChatException implements Exception {
  final int code;
  final String message;
  const ChatException(this.code, this.message);

  /// True when a registered abort predicate stopped the generation.
  bool get isAborted => code == chatErrAborted;

  @override
  String toString() => 'ChatException($code: $message)';
}

/// A registered abort predicate stopped the generation.
///
/// The session has been flushed back to its just-opened state — KV cache and
/// history cleared — so it can be reused directly with no
/// [CrispasrChatSession.reset] first.
class ChatAborted extends ChatException {
  const ChatAborted(super.code, super.message);

  @override
  String toString() => 'ChatAborted($code: $message)';
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

/// A chat session over one GGUF model.
///
/// Implements [Finalizable]: every method that passes [_handle] to C does so
/// with `this` as its receiver, and the language guarantees a `Finalizable`
/// receiver stays alive to the end of such a method body. That is what stops
/// the garbage collector running the attached `crispasr_chat_close` while a
/// native call is still blocked on the same handle.
class CrispasrChatSession implements Finalizable {
  CrispasrChatSession._(this._lib, this._closeFn, this._handle,
      this._templateName, this._nCtx, int externalSize) {
    _finalizerFor(_closeFn)
        .attach(this, _handle, detach: this, externalSize: externalSize);
  }

  final DynamicLibrary _lib;

  /// `crispasr_chat_close` in the library this session was opened from — the
  /// one pointer both [close] and the free-on-GC finalizer call.
  final Pointer<NativeFinalizerFunction> _closeFn;

  Pointer<Void> _handle;
  final String _templateName;
  final int _nCtx;
  bool _closed = false;

  /// The canonical "you are talking to an AI" disclosure (EU AI Act
  /// Art. 50(1)), read from the C ABI so it cannot drift from the CLI and
  /// server wording.
  ///
  /// **This binding is why the duty is real here.** §6.3's defence for the
  /// rest of CrispASR — "a CLI transcription tool is obvious to a reasonably
  /// well-informed person" — does not carry to a chat bubble in a Flutter app.
  /// Show this at or before the first turn, and show it *visibly*: Art. 50(5)
  /// requires disclosures to meet accessibility requirements.
  ///
  /// Art. 50(2) marking of the generated text is yours too; nothing here marks
  /// it (see `docs/eu-ai-act.md` §6.6).
  static String aiDisclosureText({DynamicLibrary? lib}) {
    lib ??= DynamicLibrary.open(CrispASR.defaultLibName());
    final fn =
        lib.lookupFunction<Pointer<Utf8> Function(), Pointer<Utf8> Function()>(
            'crispasr_chat_ai_disclosure_text');
    return fn().toDartString();
  }

  /// Name of the chat template the session resolved against
  /// (e.g. `chatml`, `llama3`, `gemma`).
  String get templateName => _templateName;

  /// Context window in tokens.
  int get nCtx => _nCtx;

  /// Address of the `crispasr_chat_close` that frees this session, on both the
  /// [close] path and the free-on-GC path.
  ///
  /// A diagnostic, not something an application needs: it exists so a test can
  /// pin that a session opened with a `libPath` override frees through the
  /// library it named, rather than through a second, independently loaded copy.
  int get closeFunctionAddress => _closeFn.address;

  // Free-on-GC backstop for a session whose owner was collected without a
  // close(). A NativeFinalizer calls a native function directly, so the
  // callback is the crispasr_chat_close of the very library this session was
  // opened from — a libPath override is honoured, and nothing re-opens a
  // library from inside the collector.
  //
  // One NativeFinalizer per close-function address, held in a static map for
  // the life of the process. A finalizer reachable only from the session it is
  // attached to could be collected in the same sweep as that session, before
  // its callback had a chance to run.
  static final Map<int, NativeFinalizer> _finalizers = {};

  static NativeFinalizer _finalizerFor(
          Pointer<NativeFinalizerFunction> close) =>
      _finalizers.putIfAbsent(close.address, () => NativeFinalizer(close));

  /// Open a chat session from a GGUF chat model on disk.
  ///
  /// Throws [UnsupportedError] when the loaded dylib doesn't expose
  /// `crispasr_chat_open` (predates the chat ABI). Throws
  /// [ChatException] when the underlying load fails (missing file,
  /// unsupported architecture, …).
  factory CrispasrChatSession.open(
    String modelPath, {
    ChatOpenParams params = const ChatOpenParams(),
    String? libPath,
  }) {
    final lib = DynamicLibrary.open(libPath ?? CrispASR.defaultLibName());
    if (!lib.providesSymbol('crispasr_chat_open')) {
      throw UnsupportedError(
        'crispasr_chat_open not found in this libcrispasr — '
        'needs CrispASR 0.8.16+ with the chat ABI.',
      );
    }

    final open = lib.lookupFunction<
        Pointer<Void> Function(
            Pointer<Utf8>, Pointer<_ChatOpenParams>, Pointer<_ChatError>),
        Pointer<Void> Function(
            Pointer<Utf8>, Pointer<_ChatOpenParams>, Pointer<_ChatError>)>(
      'crispasr_chat_open',
    );
    final tmplName = lib.lookupFunction<Pointer<Utf8> Function(Pointer<Void>),
        Pointer<Utf8> Function(Pointer<Void>)>('crispasr_chat_template_name');
    final nCtxFn = lib.lookupFunction<Int32 Function(Pointer<Void>),
        int Function(Pointer<Void>)>('crispasr_chat_n_ctx');
    // `void crispasr_chat_close(crispasr_chat_session_t)` is exactly the
    // `void f(void*)` shape NativeFinalizerFunction names, so the raw symbol
    // can be handed to NativeFinalizer with no Dart trampoline in between.
    final closeFn = lib.lookup<NativeFinalizerFunction>('crispasr_chat_close');

    // Validate before allocating anything, so a rejection leaks nothing.
    _checkNoInteriorNul(modelPath, 'modelPath');
    if (params.chatTemplate != null) {
      _checkNoInteriorNul(params.chatTemplate!, 'chatTemplate');
    }

    final paramsPtr = calloc<_ChatOpenParams>();
    final tmplPtr = _fillOpenParams(lib, params, paramsPtr);

    final pathPtr = modelPath.toNativeUtf8();
    final errPtr = calloc<_ChatError>();
    try {
      final handle = open(pathPtr, paramsPtr, errPtr);
      if (handle == nullptr) {
        throw _chatError(errPtr, 0, 'crispasr_chat_open failed');
      }
      final tp = tmplName(handle);
      final tmpl = tp == nullptr ? '' : tp.cast<Utf8>().toDartString();
      final ctx = nCtxFn(handle);
      // A GC scheduling hint only: the collector has no other way to know a
      // few words of Dart object own a multi-gigabyte model, and without it a
      // dropped session can sit uncollected for a long time. The weight file's
      // size is the honest lower bound on what the session holds.
      var externalSize = 0;
      try {
        externalSize = File(modelPath).lengthSync();
      } catch (_) {
        // Unreadable after a successful open (deleted, permissions) — the hint
        // is optional, so fall back to none rather than fail the open.
      }
      return CrispasrChatSession._(
          lib, closeFn, handle, tmpl, ctx, externalSize);
    } finally {
      calloc.free(pathPtr);
      calloc.free(errPtr);
      calloc.free(paramsPtr);
      if (tmplPtr != nullptr) calloc.free(tmplPtr);
    }
  }

  /// Conservative working set in bytes (weights + KV cache + activations) for
  /// a GGUF chat model on disk, reading its metadata but never its tensor
  /// data — a pre-flight guard for low-memory devices. [params] matters mostly
  /// for `nCtx`, which sizes the KV term linearly; leave it unset and the
  /// model's own trained context is used.
  ///
  /// The number is deliberately high, not approximate. The KV term bills both
  /// the K and the V cache at the full attention width `n_embd`, but a
  /// grouped-query model gives each layer a K/V width that is a fraction of
  /// that: on Gemma 3 1B the KV term comes out 4.50× llama.cpp's real cache
  /// (117.00 MiB against 26.00 MiB at `nCtx` 1024), which is 1.33× on the
  /// whole estimate at `nCtx` 4096. Over-reporting is the safe direction for a
  /// "will this fit?" guard: it can turn away a model that would just have
  /// fitted, and never admits one that would not.
  ///
  /// Throws [UnsupportedError] when the loaded dylib doesn't expose
  /// `crispasr_chat_memory_estimate`, [ArgumentError] for an interior NUL, and
  /// [ChatException] when the model could not be read — a model the C side
  /// could not open is a failure, not an estimate of nothing.
  static int memoryEstimate(
    String modelPath, {
    ChatOpenParams params = const ChatOpenParams(),
    String? libPath,
  }) {
    final lib = DynamicLibrary.open(libPath ?? CrispASR.defaultLibName());
    if (!lib.providesSymbol('crispasr_chat_memory_estimate')) {
      throw UnsupportedError(
        'crispasr_chat_memory_estimate not found in this libcrispasr — '
        'needs CrispASR 0.8.16+ with the chat ABI.',
      );
    }
    final estimate = lib.lookupFunction<
        Size Function(
            Pointer<Utf8>, Pointer<_ChatOpenParams>, Pointer<_ChatError>),
        int Function(Pointer<Utf8>, Pointer<_ChatOpenParams>,
            Pointer<_ChatError>)>('crispasr_chat_memory_estimate');

    // Validate before allocating anything, so a rejection leaks nothing.
    _checkNoInteriorNul(modelPath, 'modelPath');
    if (params.chatTemplate != null) {
      _checkNoInteriorNul(params.chatTemplate!, 'chatTemplate');
    }

    final paramsPtr = calloc<_ChatOpenParams>();
    final tmplPtr = _fillOpenParams(lib, params, paramsPtr);

    final pathPtr = modelPath.toNativeUtf8();
    final errPtr = calloc<_ChatError>();
    try {
      final bytes = estimate(pathPtr, paramsPtr, errPtr);
      if (bytes == 0) {
        throw _chatError(errPtr, 0, 'crispasr_chat_memory_estimate failed');
      }
      return bytes;
    } finally {
      calloc.free(pathPtr);
      calloc.free(errPtr);
      calloc.free(paramsPtr);
      if (tmplPtr != nullptr) calloc.free(tmplPtr);
    }
  }

  /// Clear the KV cache so the next generate call re-prefills from
  /// scratch. Call when starting a new conversation in a reused session.
  ///
  /// Not needed after an abort: aborting already flushes the cache and the
  /// history back to the just-opened state.
  void reset() {
    _ensureOpen();
    final reset = _lib.lookupFunction<
        Int32 Function(Pointer<Void>, Pointer<_ChatError>),
        int Function(
            Pointer<Void>, Pointer<_ChatError>)>('crispasr_chat_reset');
    final errPtr = calloc<_ChatError>();
    try {
      final rc = reset(_handle, errPtr);
      if (rc != 0) {
        throw _chatError(errPtr, rc, 'crispasr_chat_reset failed');
      }
    } finally {
      calloc.free(errPtr);
    }
  }

  /// Tokens the model's own tokenizer produces for [messages] once this
  /// session's chat template has been applied — the prompt length a *fresh*
  /// session prefills, so it can be compared straight against [nCtx] when
  /// sizing a context window.
  ///
  /// The number covers the whole prompt: the template's control tokens, the
  /// leading BOS, and the trailing generation prompt that opens the assistant
  /// turn.
  ///
  /// An empty [messages] list counts the template's own opening, which is
  /// whatever that template emits for no messages — template-dependent, and
  /// possibly nothing at all: several chat templates write only from inside
  /// their loop over the messages, and those return 0. Do not read a positive
  /// overhead into it. [generate] rejects an empty list either way, because
  /// there is nothing to prefill.
  ///
  /// A pure query: it touches neither the KV cache nor the history, so it can
  /// be called freely between generations. For a session part-way through a
  /// conversation the number is an upper bound, since the history already
  /// holds part of the prompt.
  int countTokens(List<ChatMessage> messages) {
    _ensureOpen();
    final countFn = _lib.lookupFunction<
        Int32 Function(
            Pointer<Void>, Pointer<_ChatMessage>, Size, Pointer<_ChatError>),
        int Function(Pointer<Void>, Pointer<_ChatMessage>, int,
            Pointer<_ChatError>)>('crispasr_chat_count_tokens');

    final marshalled = _marshalMessages(messages);
    final errPtr = calloc<_ChatError>();
    try {
      final n =
          countFn(_handle, marshalled.messagesPtr, messages.length, errPtr);
      if (n < 0) {
        // A negative return is a failure sentinel, not an error code — the
        // diagnostic lives in `err`.
        throw _chatError(errPtr, 0, 'crispasr_chat_count_tokens failed');
      }
      return n;
    } finally {
      calloc.free(errPtr);
      marshalled.dispose();
    }
  }

  /// One-shot generate. Applies the chat template, prefills, runs to
  /// `maxTokens` or a stop sequence, returns the assistant reply.
  ///
  /// [messages] must be the **whole conversation**, not just the new turn.
  /// The session compares the templated prompt against the tokens it already
  /// holds and decodes only what is new, so passing the full history is what
  /// makes the KV cache pay off. Passing only the latest turn is not wrong —
  /// it simply shares no prefix, so every call re-prefills from scratch.
  ///
  /// Pass [shouldContinue] to make the run cancellable. It is a *may I keep
  /// going?* predicate: return `true` to let generation continue and `false`
  /// to abort it — the polarity of `crispasr_chat_abort_callback`, forwarded
  /// to C unchanged. It runs on this isolate, synchronously, inside the
  /// native call: once before each prompt batch during prefill, once before
  /// each sampled token, and on the CPU backend from inside a running compute
  /// graph as well, so keep it cheap and non-blocking. It **must not call
  /// back into this session** — the session mutex is held for the whole
  /// generation and re-entering deadlocks. A predicate that throws aborts the
  /// generation and its exception is rethrown once the native call returns.
  ///
  /// On abort this throws [ChatAborted] and the session is flushed back to
  /// its just-opened state — KV cache and history cleared — so it can be
  /// reused with no [reset] first.
  ///
  /// Blocks the calling isolate for the duration of generation — wrap
  /// in `Isolate.run` if the host app needs the UI isolate free.
  Future<String> generate(
    List<ChatMessage> messages, {
    ChatGenerateParams params = const ChatGenerateParams(),
    bool Function()? shouldContinue,
  }) async =>
      _generateSync(messages, params, shouldContinue);

  // The whole native call, in a plain synchronous method so that `this` — a
  // Finalizable receiver — is guaranteed alive for its entire duration. The
  // public generate() is async, and Finalizable's liveness rule is stated for
  // the block that declares the variable; keeping the handle inside one
  // synchronous body puts the native call and the guarantee in the same scope
  // with nothing to reason about.
  String _generateSync(
    List<ChatMessage> messages,
    ChatGenerateParams params,
    bool Function()? shouldContinue,
  ) {
    _ensureOpen();
    final stringFree = _lib.lookupFunction<Void Function(Pointer<Utf8>),
        void Function(Pointer<Utf8>)>(
      'crispasr_chat_string_free',
    );
    final generate = _lib.lookupFunction<
        Pointer<Utf8> Function(Pointer<Void>, Pointer<_ChatMessage>, Size,
            Pointer<_ChatGenerateParams>, Pointer<_ChatError>),
        Pointer<Utf8> Function(
            Pointer<Void>,
            Pointer<_ChatMessage>,
            int,
            Pointer<_ChatGenerateParams>,
            Pointer<_ChatError>)>('crispasr_chat_generate');

    // Every rejection happens here, before a single allocation, so a bad
    // argument cannot leak what a later step would have had to free.
    for (var i = 0; i < params.stop.length; i++) {
      _checkNoInteriorNul(params.stop[i], 'stop[$i]');
    }
    _checkMessages(messages);

    final marshalled = _marshalMessages(messages);
    final marshalledParams = _marshalParams(params);
    final errPtr = calloc<_ChatError>();
    final abort = _AbortRegistration.install(_lib, _handle, shouldContinue);
    try {
      final out = generate(_handle, marshalled.messagesPtr, messages.length,
          marshalledParams.paramsPtr, errPtr);
      abort.rethrowPredicateError();
      if (out == nullptr) {
        throw _chatError(errPtr, 0, 'crispasr_chat_generate failed');
      }
      try {
        return _readNativeUtf8(out);
      } finally {
        stringFree(out);
      }
    } finally {
      abort.remove();
      calloc.free(errPtr);
      marshalledParams.dispose();
      marshalled.dispose();
    }
  }

  // generateStream is not bound on this binding yet.
  //
  // The chat C ABI passes `const char* utf8` to its on_token callback — a
  // pointer valid only for the duration of the synchronous C-side call.
  // NativeCallable.listener posts callbacks via SendPort for asynchronous
  // delivery on the owning isolate's event loop; by the time the Dart closure
  // runs, that pointer is already dangling (the C++ std::string piece has
  // gone out of scope, or worse — been re-used by a later iteration of the
  // generate loop). So .listener genuinely cannot carry a chat callback.
  //
  // NativeCallable.isolateLocal delivers synchronously on the calling thread
  // and does accept a closure — that is how the abort predicate above is
  // bound, and streaming is bindable the same way. It simply has not been
  // done yet; this is missing work, not a limitation.
  //
  // In the meantime the streaming path for Dart is the HTTP endpoint exposed
  // by `crispasr --server --chat-model …` (POST /v1/chat/completions with
  // `stream: true`), which emits SSE deltas a regular HTTP client can
  // subscribe to. The FFI path stays one-shot.

  /// Close the session and free its KV cache. Idempotent.
  ///
  /// Detaches the free-on-GC finalizer first, so the handle this call frees is
  /// not freed a second time when the object is later collected.
  ///
  /// `crispasr_chat_close` waits for calls already inside the session before
  /// it frees, which is what makes closing a busy session safe in the
  /// threaded bindings. Nothing here can reach that state: FFI calls run
  /// synchronously on the isolate that made them, so a generation and a close
  /// cannot overlap on one session. Two isolates sharing a handle would be a
  /// different story, and this class does not support that — the handle is
  /// not a transferable value.
  void close() {
    if (_closed || _handle == nullptr) return;
    _closed = true;
    _finalizerFor(_closeFn).detach(this);
    _closeFn.asFunction<void Function(Pointer<Void>)>()(_handle);
    _handle = nullptr;
  }

  void _ensureOpen() {
    if (_closed || _handle == nullptr) {
      throw StateError('CrispasrChatSession is closed');
    }
  }

  // -------------------------------------------------------------------------
  // Internal marshalling helpers
  // -------------------------------------------------------------------------

  _MarshalledParams _marshalParams(ChatGenerateParams p) {
    for (var i = 0; i < p.stop.length; i++) {
      _checkNoInteriorNul(p.stop[i], 'stop[$i]');
    }
    final ptr = calloc<_ChatGenerateParams>();
    final ref = ptr.ref;
    ref.maxTokens = p.maxTokens;
    ref.temperature = p.temperature;
    ref.topK = p.topK;
    ref.topP = p.topP;
    ref.minP = p.minP;
    ref.repeatPenalty = p.repeatPenalty;
    ref.repeatLastN = p.repeatLastN;
    ref.seed = p.seed;
    ref.prefillOnly = p.prefillOnly;
    final owned = <Pointer<Utf8>>[];
    if (p.stop.isEmpty) {
      ref.stop = nullptr;
      ref.nStop = 0;
    } else {
      // The snapshot, not the caller's list: shouldContinue runs synchronously
      // inside the native call and is free to mutate `p.stop`, so cleanup that
      // re-read its length could index past this array or leave part of it
      // allocated.
      final n = p.stop.length;
      final arr = calloc<Pointer<Utf8>>(n);
      for (var i = 0; i < n; i++) {
        arr[i] = p.stop[i].toNativeUtf8();
        owned.add(arr[i]);
      }
      ref.stop = arr;
      ref.nStop = n;
    }
    return _MarshalledParams._(ptr, ptr.ref.stop, owned);
  }

  static void _checkMessages(List<ChatMessage> messages) {
    for (var i = 0; i < messages.length; i++) {
      _checkNoInteriorNul(messages[i].role, 'messages[$i].role');
      _checkNoInteriorNul(messages[i].content, 'messages[$i].content');
    }
  }

  _MarshalledMessages _marshalMessages(List<ChatMessage> messages) {
    // Validate the whole list before allocating, so a rejection leaks nothing.
    _checkMessages(messages);
    if (messages.isEmpty) {
      return _MarshalledMessages._(nullptr, const []);
    }
    final arr = calloc<_ChatMessage>(messages.length);
    final owned = <Pointer<Utf8>>[];
    for (var i = 0; i < messages.length; i++) {
      final r = messages[i].role.toNativeUtf8();
      final c = messages[i].content.toNativeUtf8();
      owned.add(r);
      owned.add(c);
      arr[i].role = r;
      arr[i].content = c;
    }
    return _MarshalledMessages._(arr, owned);
  }
}

/// The native side of one marshalled [ChatGenerateParams], freed from what was
/// allocated rather than from what the caller's object says now.
class _MarshalledParams {
  _MarshalledParams._(this.paramsPtr, this._stopArray, this._owned);
  final Pointer<_ChatGenerateParams> paramsPtr;
  final Pointer<Pointer<Utf8>> _stopArray;
  final List<Pointer<Utf8>> _owned;
  void dispose() {
    for (final p in _owned) {
      calloc.free(p);
    }
    if (_stopArray != nullptr) {
      calloc.free(_stopArray);
    }
    calloc.free(paramsPtr);
  }
}

class _MarshalledMessages {
  _MarshalledMessages._(this.messagesPtr, this._owned);
  final Pointer<_ChatMessage> messagesPtr;
  final List<Pointer<Utf8>> _owned;
  void dispose() {
    if (messagesPtr != nullptr) {
      calloc.free(messagesPtr);
    }
    for (final p in _owned) {
      calloc.free(p);
    }
  }
}

/// One generate call's abort predicate, bound to C for the length of that
/// call and torn down after it.
///
/// `NativeCallable.isolateLocal` is what makes a Dart closure usable here:
/// it delivers on the calling thread, synchronously, which is the only way to
/// answer a predicate the native code is blocking on.
class _AbortRegistration {
  _AbortRegistration._(this._setAbort, this._session, this._callable);

  final _SetAbortDart? _setAbort;
  final Pointer<Void> _session;
  final NativeCallable<_AbortCallbackNative>? _callable;
  Object? _error;
  StackTrace? _stackTrace;

  static _AbortRegistration install(
      DynamicLibrary lib, Pointer<Void> session, bool Function()? predicate) {
    if (predicate == null) {
      return _AbortRegistration._(null, session, null);
    }
    final setAbort = lib.lookupFunction<_SetAbortNative, _SetAbortDart>(
        'crispasr_chat_set_abort_callback');
    late final _AbortRegistration reg;
    // The answer goes to C as it is: true continues, false aborts — the same
    // way round as crispasr_chat_abort_callback. Nothing inverts it.
    final callable = NativeCallable<_AbortCallbackNative>.isolateLocal(
      (Pointer<Void> _) {
        if (reg._error != null) return false;
        try {
          return predicate();
        } catch (e, st) {
          // A predicate that raised cannot be asked again: stop the run and
          // surface the failure once the native call has returned.
          reg._error = e;
          reg._stackTrace = st;
          return false;
        }
      },
      exceptionalReturn: false,
    );
    reg = _AbortRegistration._(setAbort, session, callable);
    setAbort(session, callable.nativeFunction, nullptr);
    return reg;
  }

  void rethrowPredicateError() {
    final e = _error;
    if (e != null) {
      Error.throwWithStackTrace(e, _stackTrace ?? StackTrace.current);
    }
  }

  void remove() {
    final setAbort = _setAbort;
    final callable = _callable;
    if (setAbort == null || callable == null) return;
    setAbort(_session, nullptr, nullptr);
    callable.close();
  }
}

void Function(Pointer<_ChatGenerateParams>) _generateParamsDefault(
        DynamicLibrary lib) =>
    lib.lookupFunction<Void Function(Pointer<_ChatGenerateParams>),
            void Function(Pointer<_ChatGenerateParams>)>(
        'crispasr_chat_generate_params_default');

void _checkNoInteriorNul(String value, String field) {
  if (value.codeUnits.contains(0)) {
    throw ArgumentError.value(
      value,
      field,
      'contains an interior NUL, which C cannot carry',
    );
  }
}

/// Decode a NUL-terminated native UTF-8 string permissively.
///
/// The C side appends raw detokeniser bytes, so a reply cut short by
/// `max_tokens`, a stop sequence or an abort can end mid-character. Strict
/// decoding would throw a `FormatException` out of the call instead of
/// returning the text; a replacement character is the better answer, and is
/// what the Rust and Python bindings return for the same bytes.
String _readNativeUtf8(Pointer<Utf8> p) =>
    utf8.decode(p.cast<Uint8>().asTypedList(p.length), allowMalformed: true);

/// Read the inline `char[256]` diagnostic up to the first NUL.
///
/// C fills it with `vsnprintf` output that interpolates caller data — a model
/// path, a template name — so it is UTF-8, not Latin-1, and is decoded
/// permissively for the same reason as [_readNativeUtf8].
String _readErrorMessage(Pointer<_ChatError> errPtr) {
  final ref = errPtr.ref;
  final bytes = <int>[];
  for (var i = 0; i < 256; i++) {
    final c = ref.message[i];
    if (c == 0) break;
    bytes.add(c & 0xff);
  }
  return utf8.decode(bytes, allowMalformed: true);
}

/// Turn a filled-in `crispasr_chat_error` into the right exception.
///
/// The one-shot paths signal failure by returning NULL or a negative
/// sentinel, so `err` is the only carrier there; the paths that return a code
/// pass it as [codeHint], which is used when the struct was left at zero.
ChatException _chatError(
    Pointer<_ChatError> errPtr, int codeHint, String fallback) {
  final raw = errPtr.ref.code;
  final code = raw != 0 ? raw : codeHint;
  var message = _readErrorMessage(errPtr);
  if (message.isEmpty) message = fallback;
  return code == chatErrAborted
      ? ChatAborted(code, message)
      : ChatException(code, message);
}

/// Fill [out] with the ABI's own open defaults, then apply whatever [params]
/// names. Shared by the two entry points that take open params — the session
/// factory and the pre-flight estimate — so the two cannot disagree about
/// which fields reach C.
///
/// Returns the chat-template string put into native memory, or `nullptr` when
/// [params] names none. The ABI copies the string out, so the caller frees it
/// once the call it was passed to has returned.
Pointer<Utf8> _fillOpenParams(
    DynamicLibrary lib, ChatOpenParams params, Pointer<_ChatOpenParams> out) {
  final defaults = lib.lookupFunction<
      Void Function(Pointer<_ChatOpenParams>),
      void Function(
          Pointer<_ChatOpenParams>)>('crispasr_chat_open_params_default');
  defaults(out);
  final pp = out.ref;
  if (params.nThreads != null) pp.nThreads = params.nThreads!;
  if (params.nThreadsBatch != null) pp.nThreadsBatch = params.nThreadsBatch!;
  if (params.nCtx != null) pp.nCtx = params.nCtx!;
  if (params.nBatch != null) pp.nBatch = params.nBatch!;
  if (params.nUbatch != null) pp.nUbatch = params.nUbatch!;
  if (params.nGpuLayers != null) pp.nGpuLayers = params.nGpuLayers!;
  pp.useMmap = params.useMmap;
  pp.useMlock = params.useMlock;

  final tmplPtr = params.chatTemplate != null
      ? params.chatTemplate!.toNativeUtf8()
      : nullptr;
  pp.chatTemplate = tmplPtr.cast();
  return tmplPtr;
}
