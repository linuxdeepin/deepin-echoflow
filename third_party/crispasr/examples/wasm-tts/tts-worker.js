// TTS in a Web Worker — CrispEmbed's proven pattern for MULTITHREADED Emscripten in the browser.
// Two rules that avoid the deadlock:
//   1. Pthread-pool workers (self.name === 'em-pthread') do ONLY importScripts(loader) — nothing else.
//   2. Instantiate the module at TOP LEVEL, never inside onmessage: the pthread bootstrap deadlocks
//      when the factory is first called from an active message event.
'use strict';
const LOADER = './libwhisper.js';

if (self.name === 'em-pthread') {
    importScripts(LOADER); // pthread bootstrap only
} else {
    main();
}

function main () {
    importScripts(LOADER); // defines the global `whisper_factory`

    let M = null, opened = false, loadedVoice = null, deDict = false;
    const G2P_DIR = 'home/web_user/.cache/crispasr';
    const post = (m, t) => self.postMessage(m, t || []);
    const fetchU = async u => {
        const r = await fetch(u);
        if (!r.ok) throw new Error(`HTTP ${r.status} for ${u}`);
        return new Uint8Array(await r.arrayBuffer());
    };
    const writeAt = (dir, name, bytes) => {
        try { M.FS_createPath('/', dir, true, true); } catch (e) { /* exists */ }
        M.FS_createDataFile('/' + dir, name, bytes, true, true);
    };

    // INSTANTIATE AT TOP LEVEL (not in onmessage) — this is the key to multithreaded not deadlocking.
    const ready = whisper_factory({
        print: m => post({ log: m }),
        printErr: m => post({ log: m })
    }).then(m => { M = m; post({ log: 'module ready' }); });

    self.onmessage = async (e) => {
        const { text, lang, voiceFile } = e.data;
        try {
            await ready;
            if (!opened) {
                post({ log: 'loading model…' });
                writeAt(G2P_DIR, 'cmudict.dict', await fetchU('./cmudict.dict'));
                writeAt('models', 'tts.gguf', await fetchU('./kokoro.gguf'));
                // PROXY_TO_PTHREAD build → compute runs on pthread-0, servicer stays free → real multithread.
                if (!M.ttsOpenExplicit('/models/tts.gguf', 'kokoro', 4)) throw new Error('ttsOpenExplicit failed');
                opened = true;
            }
            if (lang === 'de' && !deDict) {
                try {
                    writeAt(G2P_DIR, 'olaph_de.txt', await fetchU('./olaph_de.txt'));
                    writeAt(G2P_DIR, 'espeak_de.tsv', await fetchU('./espeak_de.tsv'));
                    deDict = true;
                } catch (err) { /* best effort */ }
            }
            if (loadedVoice !== voiceFile) {
                writeAt('models', 'voice.gguf', await fetchU('./' + voiceFile));
                M.ttsSetVoice('/models/voice.gguf', '');
                loadedVoice = voiceFile;
            }
            if (M.sessionSetSourceLanguage) M.sessionSetSourceLanguage(lang);
            post({ log: 'synthesizing…' });
            const t0 = Date.now();
            // Async: enqueue compute onto pthread-0, get the PCM via callback (servicer never blocks).
            const pcm = M.ttsSynthesizeAsync
                ? await new Promise((resolve) => M.ttsSynthesizeAsync(text, resolve))
                : M.ttsSynthesize(text);
            if (!pcm || !pcm.length) { post({ error: 'synthesis returned no audio' }); return; }
            const out = new Float32Array(pcm);
            post({ pcm: out, ms: Date.now() - t0 }, [out.buffer]);
        } catch (err) {
            post({ error: (err && err.message) || String(err) });
        }
    };

    post({ log: 'worker started' });
}
