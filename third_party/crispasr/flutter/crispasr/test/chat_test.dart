// chat_test.dart — tests for the CrispasrChatSession Dart binding.
//
// Two layers:
//   1. Symbol-resolution test (always runs): confirms every entry point of
//      include/crispasr_chat.h is exported by the loaded libcrispasr.
//   2. Real-model tests (gated on CRISPASR_CHAT_TEST_MODEL pointing at a
//      small GGUF chat model on disk): open / count_tokens / generate /
//      stop sequences / abort / reset / close end-to-end. Mirrors the
//      tests/test-chat-ggml.cpp smoke on the C++ side, and pins the same
//      stop-sequence literals the Rust, Python and Go suites pin.
//
// Run with:
//   CRISPASR_LIB=../../build-tests/src/libcrispasr.dylib \
//   CRISPASR_CHAT_TEST_MODEL=/path/to/gemma-3-1b-it-Q4_K_M.gguf \
//   dart test test/chat_test.dart

import 'dart:convert';
import 'dart:ffi';
import 'dart:io';

import 'package:crispasr/crispasr.dart';
import 'package:test/test.dart';

DynamicLibrary _openLib() {
  final path = Platform.environment['CRISPASR_LIB'];
  if (path != null && path.isNotEmpty) return DynamicLibrary.open(path);
  if (Platform.isMacOS || Platform.isIOS) {
    return DynamicLibrary.open(
      '../../build-ninja-compile/src/libcrispasr.dylib',
    );
  }
  return DynamicLibrary.open('../../build-ninja-compile/src/libcrispasr.so');
}

/// A prompt whose greedy reply is fixed and made of short, distinct pieces,
/// so a stop substring can be placed inside it and the truncated text pinned
/// exactly. The reply this model gives is "1\n2\n3\n4\n5\n6\n7\n8\n".
const _countingPrompt =
    'Count from 1 to 8. Write only the numbers, one per line, nothing else.';

/// What `_countingPrompt` yields once generation stops on "4" — the text the
/// caller receives, with the match itself cut off. The Rust, Python and Go
/// chat suites assert this same string for the same prompt, stop list and
/// sampler settings.
const _countingStoppedAtFour = '1\n2\n3\n';

/// The full greedy reply the literals above describe; isPinnedStopBaseline
/// confirms the gated model produces it before any case asserts a truncation.
const _countingBaselineReply = '1\n2\n3\n4\n5\n6\n7\n8\n';

const _countingMessages = [ChatMessage(role: 'user', content: _countingPrompt)];

/// Greedy decoding, so every case below is reproducible run to run. Every
/// other sampler field keeps its constructor default, which the parity test
/// below pins to the ABI's own default.
ChatGenerateParams _greedy(int maxTokens, {List<String> stop = const []}) =>
    ChatGenerateParams(maxTokens: maxTokens, temperature: 0.0, stop: stop);

void main() {
  late DynamicLibrary lib;

  setUpAll(() {
    lib = _openLib();
  });

  test('chat ABI symbols resolve', () {
    // Every entry point declared in include/crispasr_chat.h.
    const symbols = [
      'crispasr_chat_open_params_default',
      'crispasr_chat_generate_params_default',
      'crispasr_chat_open',
      'crispasr_chat_close',
      'crispasr_chat_reset',
      'crispasr_chat_generate',
      'crispasr_chat_generate_stream',
      'crispasr_chat_set_abort_callback',
      'crispasr_chat_template_name',
      'crispasr_chat_n_ctx',
      'crispasr_chat_count_tokens',
      'crispasr_chat_memory_estimate',
      'crispasr_chat_string_free',
      'crispasr_chat_ai_disclosure_text',
    ];
    for (final s in symbols) {
      expect(
        lib.providesSymbol(s),
        isTrue,
        reason: 'missing $s in libcrispasr',
      );
    }
  });

  test('an interior NUL in an open argument is rejected by field name', () {
    // No model needed: open() validates before it touches the filesystem, so
    // the rejection is reached whether or not a GGUF is available.
    final libPath = Platform.environment['CRISPASR_LIB'];
    expect(
      () => CrispasrChatSession.open('mo\u0000del.gguf', libPath: libPath),
      throwsA(isA<ArgumentError>().having((e) => e.name, 'name', 'modelPath')),
    );
    expect(
      () => CrispasrChatSession.open(
        'model.gguf',
        params: const ChatOpenParams(chatTemplate: 'a\u0000b'),
        libPath: libPath,
      ),
      throwsA(
        isA<ArgumentError>().having((e) => e.name, 'name', 'chatTemplate'),
      ),
    );
    // memoryEstimate takes the same strings and validates them the same way,
    // before it allocates anything.
    expect(
      () => CrispasrChatSession.memoryEstimate(
        'mo\u0000del.gguf',
        libPath: libPath,
      ),
      throwsA(isA<ArgumentError>().having((e) => e.name, 'name', 'modelPath')),
    );
  });

  test('memoryEstimate throws for a model it cannot read', () {
    // No model needed: the C side signals a failed estimate by returning 0
    // with err filled, which has to surface as an exception rather than as an
    // estimate of nothing.
    final libPath = Platform.environment['CRISPASR_LIB'];
    expect(
      () => CrispasrChatSession.memoryEstimate(
        '/nonexistent/crispasr-memory-estimate.gguf',
        libPath: libPath,
      ),
      throwsA(isA<ChatException>()),
    );
  });

  test('the error message is decoded as UTF-8, not Latin-1', () {
    // No model needed: the load fails and C formats the path it was given
    // into the char[256] with vsnprintf, so a non-ASCII path comes back
    // through the diagnostic. Decoding it byte-per-code-point would turn
    // every multi-byte character into mojibake.
    final libPath = Platform.environment['CRISPASR_LIB'];
    const path = '/tmp/crispasr-naïve-模型-нет.gguf';
    expect(
      () => CrispasrChatSession.open(path, libPath: libPath),
      throwsA(
        isA<ChatException>()
            .having((e) => e.message, 'message', contains(path))
            .having((e) => e.isAborted, 'isAborted', isFalse),
      ),
    );
  });

  test("ChatGenerateParams' defaults are the ABI's own defaults", () {
    // No model needed — crispasr_chat_generate_params_default is a pure
    // library call, so this runs on any checkout with a build.
    //
    // ChatGenerateParams carries the sampler defaults as Dart constants
    // rather than reading them from C, because reading them would make every
    // field nullable and break callers that read one back. That leaves one
    // hazard: an edit to crispasr_chat_generate_params_default in
    // src/chat.cpp would leave these constants behind, and a
    // partially-filled params object would quietly sample differently from
    // the ABI's stated defaults. This case is the guard — it fails on such an
    // edit and forces a deliberate update here.
    final libPath = Platform.environment['CRISPASR_LIB'];
    final abi = ChatGenerateParams.abiDefaults(libPath: libPath);
    const dart = ChatGenerateParams();

    expect(dart.maxTokens, equals(abi.maxTokens));
    // The C fields are float, so the Dart double literal is compared with a
    // tolerance wide enough for the float round-trip and nothing more.
    expect(dart.temperature, closeTo(abi.temperature, 1e-6));
    expect(dart.topK, equals(abi.topK));
    expect(dart.topP, closeTo(abi.topP, 1e-6));
    expect(dart.minP, closeTo(abi.minP, 1e-6));
    expect(dart.repeatPenalty, closeTo(abi.repeatPenalty, 1e-6));
    expect(dart.repeatLastN, equals(abi.repeatLastN));
    expect(dart.seed, equals(abi.seed));
    expect(dart.prefillOnly, equals(abi.prefillOnly));
    expect(dart.stop, equals(abi.stop));

    // Positive control: abiDefaults really read the library rather than
    // handing back a default-constructed object, which would make every
    // comparison above compare a value with itself. These are the numbers
    // include/crispasr_chat.h documents.
    expect(abi.maxTokens, equals(256));
    expect(abi.temperature, closeTo(0.8, 1e-6));
    expect(abi.topK, equals(40));
    expect(abi.topP, closeTo(0.95, 1e-6));
    expect(abi.minP, closeTo(0.05, 1e-6));
    expect(abi.repeatPenalty, closeTo(1.1, 1e-6));
    expect(abi.repeatLastN, equals(64));
    expect(abi.seed, equals(0));
    expect(abi.prefillOnly, isFalse);
    expect(abi.stop, isEmpty);
  });

  group('real model', () {
    final modelPath = Platform.environment['CRISPASR_CHAT_TEST_MODEL'];
    final libPath = Platform.environment['CRISPASR_LIB'];

    CrispasrChatSession openSession() {
      final s = CrispasrChatSession.open(
        modelPath!,
        params: const ChatOpenParams(
          nCtx: 2048,
          nBatch: 256,
          nUbatch: 256,
          nGpuLayers: -1,
        ),
        libPath: libPath,
      );
      addTearDown(s.close);
      return s;
    }

    /// Skip the caller unless the gated model is the one the literals below
    /// describe.
    ///
    /// Those literals are one MODEL's greedy reply, not a property of the stop
    /// feature, while the gate accepts any small chat GGUF. On
    /// smollm2-360m-instruct this prompt answers "1 2 3 4 " with SPACES, so the
    /// literal cases failed on the separator while every behavioural assertion
    /// beside them passed — a red for a reason unrelated to the code under
    /// test. The cross-binding oracle is worth keeping (Rust, Python, Go and
    /// Java pin the same strings), so check the precondition they encode
    /// instead of weakening them. Called only by the cases that assert an exact
    /// string, so the model-independent ones still run on any gate model.
    Future<bool> isPinnedStopBaseline(CrispasrChatSession s) async {
      final baseline = await s.generate(_countingMessages, params: _greedy(64));
      s.reset();
      if (baseline == _countingBaselineReply) return true;
      markTestSkipped(
        'literal stop-sequence assertions are pinned to the model whose greedy '
        'reply is ${jsonEncode(_countingBaselineReply)} (e.g. gemma-3-1b-it-Q4_K_M); '
        'this model replies ${jsonEncode(baseline)}. Behaviour is covered '
        'model-independently by tests/test-chat-ggml.cpp.',
      );
      return false;
    }

    setUp(() {
      if (modelPath == null || modelPath.isEmpty) {
        markTestSkipped('CRISPASR_CHAT_TEST_MODEL not set');
      }
    });

    test(
      'memoryEstimate covers the weights and scales with the context',
      () {
        if (modelPath == null || modelPath.isEmpty) return;
        final fileSize = File(modelPath).lengthSync();
        expect(fileSize, greaterThan(0));

        int at(int nCtx) => CrispasrChatSession.memoryEstimate(
              modelPath,
              params: ChatOpenParams(nCtx: nCtx),
              libPath: libPath,
            );

        // No nCtx: the model's own trained context sizes the KV term.
        expect(
          CrispasrChatSession.memoryEstimate(modelPath, libPath: libPath),
          greaterThan(fileSize),
        );

        final at1k = at(1024), at2k = at(2048), at4k = at(4096);
        expect(at1k, greaterThan(fileSize));
        expect(at2k, greaterThan(at1k));
        expect(at4k, greaterThan(at2k));

        // The KV term is linear in nCtx, so doubling the context doubles the
        // amount by which the estimate grows. A load path that returned before
        // reading the context / layer / embedding metadata would leave every
        // difference at zero and still report success.
        expect(
          at4k - at2k,
          equals(2 * (at2k - at1k)),
          reason: 'KV term not linear in nCtx: $at1k / $at2k / $at4k',
        );

        // Everything outside the KV term is context-independent, so back it out
        // and the remainder still has to cover the weights on disk.
        expect(at1k - (at2k - at1k), greaterThan(fileSize));
      },
      timeout: const Timeout(Duration(minutes: 3)),
    );

    test(
      'open / generate / reset / close',
      () async {
        if (modelPath == null || modelPath.isEmpty) return;
        final session = openSession();

        expect(session.nCtx, equals(2048));
        expect(session.templateName, isNotEmpty);

        const messages = [
          ChatMessage(
            role: 'system',
            content: 'You are a terse assistant. Answer in one word.',
          ),
          ChatMessage(role: 'user', content: 'Say hello.'),
        ];

        final reply = await session.generate(messages, params: _greedy(16));
        expect(reply, isNotEmpty);

        // A reset must be observable, so ask a *different* question first: the
        // cache then holds a prefix the second call cannot reuse. Asserting the
        // repeat of one greedy question equals itself would hold whether or not
        // the cache was cleared.
        await session.generate(const [
          ChatMessage(role: 'user', content: 'Name a colour.'),
        ], params: _greedy(16));
        session.reset();
        final afterReset = await session.generate(
          messages,
          params: _greedy(16),
        );
        expect(
          afterReset,
          equals(reply),
          reason: 'a reset session re-prefills and reproduces the reply',
        );
      },
      timeout: const Timeout(Duration(minutes: 3)),
    );

    test(
      'countTokens is positive, monotone, and counts the empty prompt',
      () async {
        if (modelPath == null || modelPath.isEmpty) return;
        final session = openSession();

        // An empty message array counts the template's own opening. Never an
        // error, whatever that template renders for no messages; for this
        // model's template the opening is a real, positive cost:
        // C entry point accepts n_messages == 0, unlike generate.
        final empty = session.countTokens(const []);
        expect(
          empty,
          greaterThan(0),
          reason: 'the template opening and BOS are still tokens',
        );

        final one = session.countTokens(_countingMessages);
        expect(one, greaterThan(empty));

        final two = session.countTokens(const [
          ChatMessage(role: 'user', content: _countingPrompt),
          ChatMessage(role: 'assistant', content: '1\n2\n3\n'),
          ChatMessage(role: 'user', content: _countingPrompt),
        ]);
        expect(two, greaterThan(one), reason: 'more conversation, more tokens');
        expect(
          two,
          lessThan(session.nCtx),
          reason: 'the count is comparable against nCtx directly',
        );

        // A pure query: it must not have consumed the context or the history.
        expect(session.countTokens(_countingMessages), equals(one));
      },
      timeout: const Timeout(Duration(minutes: 3)),
    );

    test(
      'stop sequences truncate before the match',
      () async {
        if (modelPath == null || modelPath.isEmpty) return;
        final session = openSession();
        if (!await isPinnedStopBaseline(session)) return;

        final full = await session.generate(
          _countingMessages,
          params: _greedy(64),
        );
        // Positive control. Without it the case is vacuous: a reply that never
        // reached the stop string would pass whether or not stops work at all.
        expect(
          full.contains('5'),
          isTrue,
          reason: 'the unstopped reply must contain the stop string: $full',
        );
        expect(full, equals('1\n2\n3\n4\n5\n6\n7\n8\n'));
        session.reset();

        final stopped = await session.generate(
          _countingMessages,
          params: _greedy(64, stop: ['5']),
        );
        expect(
          stopped.contains('5'),
          isFalse,
          reason: 'the matched text must not reach the caller: $stopped',
        );
        expect(full.startsWith(stopped), isTrue);
        expect(stopped, equals('1\n2\n3\n4\n'));

        // An empty stop list is the same as passing none.
        session.reset();
        final noStops = await session.generate(
          _countingMessages,
          params: _greedy(64, stop: const []),
        );
        expect(noStops, equals(full));
      },
      timeout: const Timeout(Duration(minutes: 3)),
    );

    test(
      'stop sequences stop at the earliest match in the output',
      () async {
        if (modelPath == null || modelPath.isEmpty) return;
        final session = openSession();
        if (!await isPinnedStopBaseline(session)) return;

        final full = await session.generate(
          _countingMessages,
          params: _greedy(64),
        );
        expect(
          full.contains('4') && full.contains('7'),
          isTrue,
          reason: 'the unstopped reply must contain both stop strings: $full',
        );

        // Two sequences, in both orders. "4" is generated before "7", so "4"
        // wins either way: the earliest match in the output decides, not the
        // position in the array. Order-independence is also what says the whole
        // array was marshalled, rather than only its first element.
        for (final stop in [
          ['7', '4'],
          ['4', '7'],
        ]) {
          session.reset();
          final stopped = await session.generate(
            _countingMessages,
            params: _greedy(64, stop: stop),
          );
          expect(stopped, equals(_countingStoppedAtFour), reason: 'stop $stop');
        }
      },
      timeout: const Timeout(Duration(minutes: 3)),
    );

    test(
      'the abort predicate returns true to continue, false to abort',
      () async {
        if (modelPath == null || modelPath.isEmpty) return;
        final session = openSession();
        if (!await isPinnedStopBaseline(session)) return;

        // Two-sided on purpose. Either half alone passes under an inverted
        // implementation: always-true would abort immediately (and an
        // always-false run producing text would be read as "it worked").
        var continueCalls = 0;
        final produced = await session.generate(
          _countingMessages,
          params: _greedy(64),
          shouldContinue: () {
            continueCalls++;
            return true;
          },
        );
        expect(
          continueCalls,
          greaterThan(0),
          reason: 'the predicate must actually be consulted',
        );
        expect(
          produced,
          equals('1\n2\n3\n4\n5\n6\n7\n8\n'),
          reason: 'always-continue must produce the full reply',
        );

        session.reset();
        var abortCalls = 0;
        await expectLater(
          session.generate(
            _countingMessages,
            params: _greedy(64),
            shouldContinue: () {
              abortCalls++;
              return false;
            },
          ),
          throwsA(
            isA<ChatAborted>()
                .having((e) => e.code, 'code', chatErrAborted)
                .having((e) => e.isAborted, 'isAborted', isTrue),
          ),
        );
        expect(abortCalls, greaterThan(0));
      },
      timeout: const Timeout(Duration(minutes: 3)),
    );

    test(
      'an abort stops early and leaves the session reusable without reset',
      () async {
        if (modelPath == null || modelPath.isEmpty) return;
        final session = openSession();
        if (!await isPinnedStopBaseline(session)) return;

        // How many times a run that goes all the way consults the predicate.
        // How many calls that is depends on the backend, so measure it rather
        // than pinning a number.
        var fullCalls = 0;
        final full = await session.generate(
          _countingMessages,
          params: _greedy(64),
          shouldContinue: () {
            fullCalls++;
            return true;
          },
        );
        expect(full, equals('1\n2\n3\n4\n5\n6\n7\n8\n'));
        session.reset();

        // Now stop after a handful of calls: the run must end sooner than the
        // complete one did.
        var seen = 0;
        Object? caught;
        try {
          await session.generate(
            _countingMessages,
            params: _greedy(64),
            shouldContinue: () => ++seen < 4,
          );
        } catch (e) {
          caught = e;
        }
        expect(caught, isA<ChatAborted>());
        expect(
          seen,
          lessThan(fullCalls),
          reason: 'the run stopped rather than decoding to max_tokens',
        );

        // No reset() here on purpose: an abort already flushed the KV cache and
        // the history, so the next call must prefill from scratch and produce
        // the whole reply.
        final after = await session.generate(
          _countingMessages,
          params: _greedy(64),
        );
        expect(after, equals('1\n2\n3\n4\n5\n6\n7\n8\n'));

        // The distinction a caller needs: a cancel is not a fault.
        expect((caught! as ChatException).isAborted, isTrue);
      },
      timeout: const Timeout(Duration(minutes: 3)),
    );

    test(
      'a stop list mutated from the predicate is freed from the snapshot',
      () async {
        if (modelPath == null || modelPath.isEmpty) return;
        final session = openSession();
        if (!await isPinnedStopBaseline(session)) return;

        // A growable list the caller keeps a handle on. shouldContinue runs
        // synchronously inside the native call and is forbidden only from
        // re-entering this session, so mutating this list mid-generation is
        // legal — and cleanup that re-read its length would walk off the end of
        // the native pointer array it allocated, or leave part of it behind.
        // Growing it is the case that corrupts rather than merely leaks: cleanup
        // driven by the list's CURRENT length reads two pointers past the end of
        // a two-slot native array and frees whatever it finds there.
        final stop = <String>['zzzz', 'qqqq'];
        var calls = 0;
        final reply = await session.generate(
          _countingMessages,
          params: _greedy(24, stop: stop),
          shouldContinue: () {
            if (calls++ == 0) {
              stop
                ..add('added-after-marshalling')
                ..add('and-another-one')
                ..add('and-a-fourth');
            }
            return true;
          },
        );

        // The run itself is unaffected: C read the count that was frozen into
        // the struct, and none of these appears in a count to eight.
        expect(reply, equals('1\n2\n3\n4\n5\n6\n7\n8\n'));
        expect(calls, greaterThan(0), reason: 'the predicate never ran');
        expect(stop, hasLength(5), reason: 'the mutation really happened');

        // Surviving the free is the assertion. Generate again so the failure
        // mode is a live one rather than something the process carries to exit.
        session.reset();
        expect(
          await session.generate(_countingMessages, params: _greedy(24)),
          equals('1\n2\n3\n4\n5\n6\n7\n8\n'),
        );
      },
      timeout: const Timeout(Duration(minutes: 3)),
    );

    test(
      'an interior NUL is rejected by field name, never truncated',
      () async {
        if (modelPath == null || modelPath.isEmpty) return;
        final session = openSession();

        expect(
          () => session.countTokens(const [
            ChatMessage(role: 'us\u0000er', content: 'hi'),
          ]),
          throwsA(
            isA<ArgumentError>().having(
              (e) => e.name,
              'name',
              'messages[0].role',
            ),
          ),
        );
        expect(
          () => session.countTokens(const [
            ChatMessage(role: 'user', content: 'a\u0000b'),
          ]),
          throwsA(
            isA<ArgumentError>().having(
              (e) => e.name,
              'name',
              'messages[0].content',
            ),
          ),
        );
        await expectLater(
          session.generate(
            _countingMessages,
            params: const ChatGenerateParams(stop: ['ok', 'a\u0000b']),
          ),
          throwsA(
            isA<ArgumentError>().having((e) => e.name, 'name', 'stop[1]'),
          ),
        );
      },
      timeout: const Timeout(Duration(minutes: 3)),
    );

    test(
      'a partially-filled generate params object decodes as the ABI does',
      () async {
        if (modelPath == null || modelPath.isEmpty) return;
        final session = openSession();

        // The parity case above pins the numbers; this pins that generate
        // really marshals them — a params object naming only max_tokens has to
        // reach C as a usable sampler configuration.
        final fromPartial = await session.generate(
          _countingMessages,
          params: const ChatGenerateParams(maxTokens: 24),
        );
        expect(fromPartial, isNotEmpty);
      },
      timeout: const Timeout(Duration(minutes: 3)),
    );

    test(
      'a reply cut mid-character is returned, not thrown',
      () async {
        if (modelPath == null || modelPath.isEmpty) return;
        final session = openSession();
        // Depends on the model actually emitting this character, which is as
        // model-specific as the counting literals above.
        if (!await isPinnedStopBaseline(session)) return;

        // The model spells these with byte-fallback tokens — one token per
        // UTF-8 byte — so a max_tokens cut can land inside a character. The C
        // side hands back the raw bytes, and a strict decode would throw a
        // FormatException out of generate() instead of returning the text.
        const prompt =
            'Reply with exactly this and nothing else: \u{1FABF}\u{1FACF}\u{1FABC}';
        const messages = [ChatMessage(role: 'user', content: prompt)];

        // Positive control: a cut on a character boundary keeps the character
        // whole, so the replacement below is truncation and not a decode that
        // mangles everything.
        final whole = await session.generate(messages, params: _greedy(4));
        expect(whole, equals('\u{1FABF}'));

        session.reset();
        final cut = await session.generate(messages, params: _greedy(6));
        expect(
          cut,
          equals('\u{1FABF}\uFFFD'),
          reason: 'the incomplete trailing sequence becomes U+FFFD',
        );
      },
      timeout: const Timeout(Duration(minutes: 3)),
    );

    test(
      'close frees once, is idempotent, and a closed session throws',
      () async {
        if (modelPath == null || modelPath.isEmpty) return;
        final session = openSession();

        // Positive control: the session works before the close, so the
        // StateErrors below are the close talking rather than a session that
        // never opened properly.
        expect(session.countTokens(_countingMessages), greaterThan(0));

        session.close();
        // Repeated closes must not reach crispasr_chat_close a second time. A
        // double free would take the whole process down, not fail an
        // expectation — the test surviving to its end is the assertion.
        session.close();
        session.close();
        // The teardown registered by openSession closes it once more.

        // Use after close is a clean Dart error, not a call on a dangling
        // pointer.
        expect(() => session.countTokens(_countingMessages), throwsStateError);
        expect(session.reset, throwsStateError);
        await expectLater(
          session.generate(_countingMessages, params: _greedy(8)),
          throwsStateError,
        );
      },
      timeout: const Timeout(Duration(minutes: 3)),
    );

    test(
      'a libPath override frees through the library it named',
      () {
        if (modelPath == null || modelPath.isEmpty) return;
        if (libPath == null || libPath.isEmpty) {
          markTestSkipped('CRISPASR_LIB not set — nothing to override with');
          return;
        }

        // A copy of libcrispasr in a temp directory is a separate file, so dyld
        // loads it as a second image: its crispasr_chat_close sits at a
        // different address from the one the rest of this suite uses. That is
        // what makes the assertions below able to tell the two libraries apart.
        final dir = Directory.systemTemp.createTempSync('crispasr-libcopy');
        addTearDown(() => dir.deleteSync(recursive: true));
        final copyPath = '${dir.path}/libcrispasr-copy.dylib';
        File(libPath).copySync(copyPath);

        final primaryClose = DynamicLibrary.open(
          libPath,
        ).lookup<NativeFinalizerFunction>('crispasr_chat_close');
        final copyClose = DynamicLibrary.open(
          copyPath,
        ).lookup<NativeFinalizerFunction>('crispasr_chat_close');
        // Control: without two distinct addresses every comparison below would
        // hold no matter which library a session picked.
        expect(
          copyClose.address,
          isNot(equals(primaryClose.address)),
          reason: 'the copy must load as an independent image',
        );

        const tiny = ChatOpenParams(
          nCtx: 256,
          nBatch: 64,
          nUbatch: 64,
          nGpuLayers: 0,
          useMmap: true,
        );

        final viaCopy = CrispasrChatSession.open(
          modelPath,
          params: tiny,
          libPath: copyPath,
        );
        addTearDown(viaCopy.close);
        expect(
          viaCopy.closeFunctionAddress,
          equals(copyClose.address),
          reason: 'the override names the library that must free the session',
        );
        expect(
          viaCopy.closeFunctionAddress,
          isNot(equals(primaryClose.address)),
        );

        final viaPrimary = CrispasrChatSession.open(
          modelPath,
          params: tiny,
          libPath: libPath,
        );
        addTearDown(viaPrimary.close);
        expect(viaPrimary.closeFunctionAddress, equals(primaryClose.address));
      },
      timeout: const Timeout(Duration(minutes: 3)),
    );

    test(
      'a session dropped without close is freed on garbage collection',
      () {
        if (modelPath == null || modelPath.isEmpty) return;

        // useMmap: false puts the weights on the heap, so a session that is
        // never freed shows up as resident memory that never comes back;
        // nGpuLayers: 0 keeps them there rather than on the GPU.
        const heapResident = ChatOpenParams(
          nCtx: 256,
          nBatch: 64,
          nUbatch: 64,
          nGpuLayers: 0,
          useMmap: false,
        );

        // Open a session and drop the only reference to it, without closing.
        void openAndDrop() {
          final s = CrispasrChatSession.open(
            modelPath,
            params: heapResident,
            libPath: libPath,
          );
          expect(s.nCtx, greaterThan(0));
        }

        final modelBytes = File(modelPath).lengthSync();
        final start = ProcessInfo.currentRss;
        var peak = start;
        const rounds = 5;
        for (var i = 0; i < rounds; i++) {
          openAndDrop();
          // Dart heap churn, so the collector has a reason to run. The
          // externalSize the session reports is what makes it worth its while.
          var junk = <List<int>>[];
          for (var j = 0; j < 200; j++) {
            junk.add(List<int>.filled(20000, j));
          }
          junk = const [];
          if (ProcessInfo.currentRss > peak) peak = ProcessInfo.currentRss;
        }
        final growth = ProcessInfo.currentRss - start;

        // Control: the rounds really did load weights into memory, so the bound
        // below is measuring something. Without this a run where every open
        // somehow cost nothing would pass while proving nothing.
        expect(
          peak - start,
          greaterThan(modelBytes),
          reason: 'the sessions must have been resident to begin with',
        );

        // $rounds sessions never freed would sit at roughly $rounds model
        // copies; freed on collection they plateau at about two. Four is well
        // clear of both.
        expect(
          growth,
          lessThan(4 * modelBytes),
          reason: 'dropped sessions must be freed rather than accumulate: '
              '${growth ~/ (1 << 20)} MiB after $rounds rounds of '
              '${modelBytes ~/ (1 << 20)} MiB each',
        );
      },
      timeout: const Timeout(Duration(minutes: 3)),
    );

    test(
      'prefillOnly prefills without generating',
      () async {
        if (modelPath == null || modelPath.isEmpty) return;
        final session = openSession();

        final quiet = await session.generate(
          _countingMessages,
          params: const ChatGenerateParams(
            maxTokens: 64,
            temperature: 0.0,
            prefillOnly: true,
          ),
        );
        expect(
          quiet,
          isEmpty,
          reason: 'prefill_only suppresses assistant generation',
        );

        // Positive control: the same prompt without the flag does generate, so
        // an empty reply above cannot be an empty reply in general.
        session.reset();
        final loud = await session.generate(
          _countingMessages,
          params: _greedy(64),
        );
        expect(loud, isNotEmpty);
      },
      timeout: const Timeout(Duration(minutes: 3)),
    );
  });
}
