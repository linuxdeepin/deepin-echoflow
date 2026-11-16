// crispasr_voice_provenance.h — the IO half of the voice-clone gate.
//
// Resolves a --voice / "voice" value to the file the backend will actually
// open, reads the provenance stamp out of a baked pack, and hands both to the
// pure predicate in crispasr_voice_clone_policy.h.
//
// Split from the policy header on purpose: the policy stays weight-free and
// IO-free so tests/test-voice-clone-policy.cpp can exercise it with no GGUF, no
// filesystem and no link against crispasr-core. Everything that needs a file
// lives here.

#pragma once

#include "core/gguf_loader.h"
#include "crispasr_speaker_identity_models.h" // identity_for_voice_pack — legacy pack fallback
#include "crispasr_voice_clone_policy.h"

#include <filesystem>
#include <map>
#include <mutex>
#include <string>

namespace crispasr_voice {

// True if `path` is a baked pack that declares it was derived from a real
// recording. False for presets, for unstamped packs, and for anything
// unreadable — an unreadable pack is not evidence of a clone, and the backend
// is about to fail on it anyway.
struct PackProvenance {
    // The pack carries the baker's crispasr.voice.cloned_from_recording stamp.
    bool declares_clone = false;
    // general.architecture — names the producer, used to classify LEGACY packs
    // that predate the stamp. Empty when unreadable.
    std::string architecture;
    // Whose voice this is (crispasr.voice.speaker_identity). INDEPENDENT of
    // declares_clone: a pack can be a non-clone preset and still be a real
    // person, which is the whole reason this field exists.
    SpeakerIdentity identity = SpeakerIdentity::Unknown;
};

// Read both provenance signals in ONE metadata open. Returns an empty result
// for non-packs, missing files, and anything unreadable — an unreadable pack is
// not evidence of a clone, and the backend is about to fail on it anyway.
inline PackProvenance read_pack_provenance(const std::string& path) {
    PackProvenance p;
    if (path.empty() || !is_voice_pack(path))
        return p;
    std::error_code ec;
    if (!std::filesystem::exists(path, ec))
        return p;
    gguf_context* meta = core_gguf::open_metadata(path.c_str());
    if (!meta)
        return p;
    p.declares_clone = core_gguf::kv_bool(meta, provenance_key(), false);
    p.architecture = core_gguf::kv_str(meta, "general.architecture", "");
    p.identity = parse_speaker_identity(core_gguf::kv_str(meta, speaker_identity_key(), ""));
    core_gguf::free_metadata(meta);
    return p;
}

inline bool pack_declares_clone(const std::string& path) {
    return read_pack_provenance(path).declares_clone;
}

// Read what a multi-voice BANK says about one entry.
//
// A bank is a single GGUF holding many voices, selected by name (cosyvoice3's
// voices.gguf). The gate never sees its path — the backend discovers it as a
// sibling of the model or from an env var — so it has to be handed in; see
// CrispasrBackend::voice_bank_path().
//
// Precedence: the per-voice stamp, then the bank-wide one. `has_stamps` reports
// whether the bundle carries provenance metadata at all, which is what
// distinguishes "this entry is explicitly not a clone" from "this bundle
// predates the stamp and cannot say" — only the latter falls back to the
// producer architecture.
inline crispasr_voice::BankFacts read_bank_provenance(const std::string& bank_path, const std::string& voice_name) {
    crispasr_voice::BankFacts f;
    if (bank_path.empty() || voice_name.empty())
        return f;
    std::error_code ec;
    if (!std::filesystem::exists(bank_path, ec))
        return f;
    gguf_context* meta = core_gguf::open_metadata(bank_path.c_str());
    if (!meta)
        return f;
    f.architecture = core_gguf::kv_str(meta, "general.architecture", "");
    // Sentinel written by every current baker: "this bundle stamps its entries",
    // so an absent per-voice key means "not a clone" rather than "unknown".
    f.has_stamps = core_gguf::kv_bool(meta, bank_stamped_key(), false);
    const std::string per_voice = bank_provenance_key_for(voice_name);
    f.declares_clone =
        core_gguf::kv_bool(meta, per_voice.c_str(), false) || core_gguf::kv_bool(meta, provenance_key(), false);
    // Per-entry identity, else the bank-wide one. Same precedence as the clone
    // stamp above, for the same reason: one bundle, voices of differing status.
    const std::string per_voice_identity = speaker_identity_key_for(voice_name);
    f.identity = parse_speaker_identity(core_gguf::kv_str(meta, per_voice_identity.c_str(), ""));
    if (f.identity == SpeakerIdentity::Unknown)
        f.identity = parse_speaker_identity(core_gguf::kv_str(meta, speaker_identity_key(), ""));
    core_gguf::free_metadata(meta);
    return f;
}

// Whose voice does this MODEL declare it produces?
//
// The durable half of the identity answer, and the one that retires the
// file-name matching in crispasr_speaker_identity_models.h: a checkpoint that
// carries `crispasr.voice.speaker_identity` says so itself and survives being
// renamed, re-quantised or moved. The converters write it (see
// models/convert-*.py --speaker-identity); the table stays as the legacy
// fallback for checkpoints published before the stamp existed, exactly as
// architecture_is_recording_derived() does for unstamped voice packs.
//
// MEMOISED, because this is called per synthesis request on the server and a
// GGUF metadata open per request on a hot path is the kind of thing that gets
// noticed as a latency regression long before anyone connects it to a
// compliance gate. Keyed by path; a model file does not change identity under a
// running process.
inline SpeakerIdentity read_model_speaker_identity(const std::string& model_path) {
    if (model_path.empty())
        return SpeakerIdentity::Unknown;
    struct Cache {
        std::mutex mu;
        std::map<std::string, SpeakerIdentity> seen;
    };
    static Cache cache;
    {
        std::lock_guard<std::mutex> lock(cache.mu);
        auto it = cache.seen.find(model_path);
        if (it != cache.seen.end())
            return it->second;
    }
    SpeakerIdentity id = SpeakerIdentity::Unknown;
    std::error_code ec;
    if (std::filesystem::exists(model_path, ec)) {
        if (gguf_context* meta = core_gguf::open_metadata(model_path.c_str())) {
            id = parse_speaker_identity(core_gguf::kv_str(meta, speaker_identity_key(), ""));
            core_gguf::free_metadata(meta);
        }
    }
    std::lock_guard<std::mutex> lock(cache.mu);
    cache.seen[model_path] = id;
    return id;
}

// The `general.architecture` a GGUF declares about itself, or empty.
//
// Used by --print-speaker-identity to pick the right verdict table for a file
// it was handed with no session attached. Cheap: metadata open, one string.
inline std::string read_gguf_architecture(const std::string& path) {
    if (path.empty())
        return {};
    std::error_code ec;
    if (!std::filesystem::exists(path, ec))
        return {};
    gguf_context* meta = core_gguf::open_metadata(path.c_str());
    if (!meta)
        return {};
    std::string arch = core_gguf::kv_str(meta, "general.architecture", "");
    core_gguf::free_metadata(meta);
    return arch;
}

// Resolve the value a caller passed to the file a backend will open.
//
// An absolute/relative path is returned as-is. A BARE name (the server's
// documented form, e.g. {"voice": "victim"}) is resolved against `voice_dir`
// the way the TTS adapters resolve it — pack first, then recording. Resolving
// here is load-bearing, not cosmetic: the gate used to inspect the raw string,
// so a bare name reached the same .wav while scoring as "not a clone".
inline std::string resolve_voice_path(const std::string& voice, const std::string& voice_dir) {
    if (voice.empty() || voice_dir.empty())
        return voice;
    // Already a path or an explicit file — nothing to resolve.
    if (is_voice_pack(voice) || is_recording_reference(voice))
        return voice;
    if (voice.find('/') != std::string::npos || voice.find('\\') != std::string::npos)
        return voice;
    // Reject traversal before touching the filesystem (mirrors the adapters).
    if (voice.find("..") != std::string::npos || voice.find('\0') != std::string::npos)
        return voice;
    std::error_code ec;
    const std::string gguf = voice_dir + "/" + voice + ".gguf";
    if (std::filesystem::exists(gguf, ec))
        return gguf;
    const std::string wav = voice_dir + "/" + voice + ".wav";
    if (std::filesystem::exists(wav, ec))
        return wav;
    return voice;
}

// The whole gate in one call: resolve, read provenance, classify.
//
// `baked_from_wav_this_run` is the runtime's own knowledge that it baked this
// voice from a user recording during this run — it outranks anything the file
// says, and is the reason the TADA inline clone is gated at all.
//
// `bank_path` is the multi-voice bundle the backend will select from, when it
// has one (CrispasrBackend::voice_bank_path()). Without it a bank entry is a
// bare name that names no file, and the whole gate silently returns "preset" —
// which is how cosyvoice3's voice-clone bundles went ungated on every surface.
inline CloneDecision classify_voice(const std::string& voice, const std::string& voice_dir,
                                    bool baked_from_wav_this_run, const std::string& bank_path = std::string()) {
    const std::string resolved = resolve_voice_path(voice, voice_dir);
    // Skip the GGUF read when the answer can't depend on it — avoids opening a
    // file per request on the server's hot path for presets passed by name.
    if (baked_from_wav_this_run || is_recording_reference(resolved)) {
        // A raw recording carries no metadata to declare an identity, and does
        // not need one: it is a clone, so it is disclosed and gated regardless.
        return classify(resolved, baked_from_wav_this_run, /*pack_declares_clone=*/false);
    }
    const PackProvenance p = read_pack_provenance(resolved);
    // Only a name that resolved to no file can be a bank entry. A real path was
    // already answered for by the pack read above, and asking the bank about it
    // would let an unrelated bundle's metadata classify someone else's file.
    const bool is_bank_entry = !bank_path.empty() && !is_voice_pack(resolved) && !is_recording_reference(resolved);
    const BankFacts bank = is_bank_entry ? read_bank_provenance(bank_path, resolved) : BankFacts();
    CloneDecision d = classify(resolved, baked_from_wav_this_run, p.declares_clone, p.architecture, bank);
    // Whichever source actually described this voice is the one that can speak
    // for its identity: a bank entry's own key, else the pack's.
    d.pack_identity = is_bank_entry ? bank.identity : p.identity;
    // Legacy fallback for packs published before the stamp existed — the same
    // role the model table plays for checkpoints. Only consulted when the pack
    // said nothing; a stamped pack always answers for itself.
    //
    // This is where kokoro's answer lives: its checkpoint is a backbone, not a
    // voice, so `--voice kokoro-voice-df_eva.gguf` is the only thing that knows
    // you are about to hear a specific HUI narrator.
    if (d.pack_identity == SpeakerIdentity::Unknown)
        d.pack_identity = identity_for_voice_pack(resolved);
    return d;
}

} // namespace crispasr_voice
