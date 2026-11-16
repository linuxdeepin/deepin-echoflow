# Java JNI bindings for CrispASR

This package provides Java JNI bindings for crispasr. They have been tested on:

  * <strike>Darwin (OS X) 12.6 on x64_64</strike>
  * Ubuntu on x86_64
  * Windows on x86_64

The "low level" bindings are in `WhisperCppJnaLibrary`. The most simple usage is as follows:

JNA will attempt to load the `whispercpp` shared library from:

- jna.library.path
- jna.platform.library
- ~/Library/Frameworks
- /Library/Frameworks
- /System/Library/Frameworks
- classpath

```java
import io.github.ggerganov.whispercpp.WhisperCpp;

public class Example {

    public static void main(String[] args) {
        
        WhisperCpp whisper = new WhisperCpp();
        try {
            // By default, models are loaded from ~/.cache/whisper/ and are usually named "ggml-${name}.bin"
            // or you can provide the absolute path to the model file.
            whisper.initContext("../ggml-base.en.bin"); 
            WhisperFullParams.ByValue whisperParams = whisper.getFullDefaultParams(WhisperSamplingStrategy.CRISPASR_SAMPLING_BEAM_SEARCH); 
            
            // custom configuration if required      
            //whisperParams.n_threads = 8;
            whisperParams.temperature = 0.0f;
            whisperParams.temperature_inc = 0.2f;
            //whisperParams.language = "en";
                            
            float[] samples = readAudio(); // divide each value by 32767.0f
            List<WhisperSegment> whisperSegmentList = whisper.fullTranscribeWithTime(whisperParams, samples);
            
            for (WhisperSegment whisperSegment : whisperSegmentList) {

                long start = whisperSegment.getStart();
                long end = whisperSegment.getEnd();

                String text = whisperSegment.getSentence();
                    
                System.out.println("start: "+start);
                System.out.println("end: "+end);
                System.out.println("text: "+text);
                
            }
    
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            whisper.close();
        }
        
     }
}
```

## Chat / LLM

`io.github.ggerganov.whispercpp.chat` binds the `crispasr_chat_*` C ABI
(`include/crispasr_chat.h`) — text in, text out, separate from the ASR surface
above and usable on its own.

```java
import io.github.ggerganov.whispercpp.chat.*;
import java.util.Arrays;
import java.util.List;

List<ChatMessage> turns = Arrays.asList(
        ChatMessage.system("You are terse."),
        ChatMessage.user("Name three primes."));

try (ChatSession s = ChatSession.open("/models/gemma-3-1b-it-Q4_K_M.gguf",
        new ChatOpenParams().nCtx(4096))) {

    System.out.println(s.templateName() + ", " + s.nCtx() + " tokens");
    System.out.println(s.countTokens(turns) + " tokens of prompt");

    // One shot.
    System.out.println(s.generate(turns, new ChatGenerateParams().maxTokens(128)));

    // Or streamed. Chunks are whole characters; concatenated they equal the
    // one-shot text. The listener must not call back into the same session.
    s.generateStream(turns, new ChatGenerateParams().maxTokens(128), System.out::print);
}
```

Points worth knowing:

- **Pass the WHOLE conversation on every call.** The session compares the
  templated prompt against the tokens it already holds and decodes only what is
  new; passing just the latest turn re-prefills from scratch.
- **Cancellation.** `setAbortCallback` takes a predicate that returns **true to
  CONTINUE** — the polarity of the C callback and of the ASR side's
  encoder-begin callback. A cancelled call throws `ChatAbortedException` and
  leaves the session as if freshly opened, so no `reset()` is needed.
- **Params keep the ABI defaults.** `new ChatGenerateParams().maxTokens(64)`
  leaves `temperature` at 0.8, not 0.0.
- **`maxTokens(0)`** selects the ABI default of 256, not "generate nothing" —
  use `prefillOnly(true)` for that.
- **`memoryEstimate` over-reports on purpose.** It bills the KV cache at the
  full attention width, so on a grouped-query model the figure comes out well
  above the real working set — the safe direction for a "will this fit?"
  pre-flight guard. See its javadoc for the measured factor.
- **EU AI Act Art. 50.** This is synthetic text generation and nothing marks it.
  `ChatSession.aiDisclosureText()` is the canonical "you are talking to an AI"
  wording; show it visibly, and mark the output machine-readably yourself.

The end-to-end cases in `ChatSessionTest` need a GGUF chat model and are gated
on `CRISPASR_CHAT_TEST_MODEL`, self-skipping when it is unset:

```bash
CRISPASR_CHAT_TEST_MODEL=/models/gemma-3-1b-it-Q4_K_M.gguf ./gradlew test
```

## Building & Testing

In order to build, you need to have the JDK 8 or higher installed. Run the tests with:

```bash
git clone https://github.com/ggml-org/crispasr.git
cd crispasr/bindings/java

./gradlew build
```

You need to have the `crispasr` library in your [JNA library path](https://java-native-access.github.io/jna/4.2.1/com/sun/jna/NativeLibrary.html). On Windows the dll is included in the jar and you can update it:

```bash
copy /y ..\..\build\bin\Release\crispasr.dll build\generated\resources\main\win32-x86-64\crispasr.dll
```


## License

The license for the Java bindings is the same as the license for the rest of the crispasr project, which is the MIT License. See the `LICENSE` file for more details.
