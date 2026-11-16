// separation_io.h — shared source-separation output surface (§248).
//
// One place that BOTH separation backends (htdemucs 4-stem, mel-band-roformer
// vocal/instrumental — and any future one) route their stem output through, so
// the CLI/bindings/server don't each reinvent naming + WAV writing. The audio
// graph lives in each backend; this header owns only the backend-agnostic
// RESULT VIEW and the file-output policy.
//
// Design (see docs/source-separation-surface.md): a backend produces N named
// stems as interleaved float32; the dispatcher wraps them in a non-owning
// `crispasr_separation_view` and this header turns each selected stem into a
// deterministic output path + a stereo/mono WAV. Header-only so the naming
// policy is unit-testable without linking a backend.

#pragma once

#include "crispasr_wav_writer.h" // crispasr_make_wav_int16_interleaved

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

// Non-owning view of a separation result. `sources[s]` points at
// n_channels * n_frames interleaved float32 samples; the backend owns the
// storage. `source_names[s]` is a stable lowercase label ("vocals", "drums").
struct crispasr_separation_view {
    int n_sources = 0;
    int n_channels = 0;
    int n_frames = 0; // per channel
    int sample_rate = 0;
    const float* const* sources = nullptr;
    const char* const* source_names = nullptr;
};

// Lowercase a copy (ASCII).
inline std::string crispasr_sep_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}

// Split the input path into (directory-with-trailing-slash-or-empty, basename
// without its final extension). Handles both '/' and '\\' separators. A
// leading-dot name (".env") is treated as having no extension.
inline void crispasr_sep_split_path(const std::string& path, std::string& dir, std::string& stem) {
    const size_t slash = path.find_last_of("/\\");
    dir = (slash == std::string::npos) ? "" : path.substr(0, slash + 1);
    const std::string base = (slash == std::string::npos) ? path : path.substr(slash + 1);
    const size_t dot = base.find_last_of('.');
    stem = (dot == std::string::npos || dot == 0) ? base : base.substr(0, dot);
}

// Output path for one stem: "<out_dir>/<input-stem>_<source>.<ext>". When
// out_dir is empty the stem is written alongside the input. out_dir may or may
// not carry a trailing slash. Example:
//   crispasr_stem_output_path("/m/song.flac", "vocals", "", "wav")
//     -> "/m/song_vocals.wav"
inline std::string crispasr_stem_output_path(const std::string& input_path, const std::string& source_name,
                                             const std::string& out_dir, const std::string& ext = "wav") {
    std::string in_dir, stem;
    crispasr_sep_split_path(input_path, in_dir, stem);
    std::string dir = out_dir.empty() ? in_dir : out_dir;
    if (!dir.empty() && dir.back() != '/' && dir.back() != '\\')
        dir.push_back('/');
    return dir + stem + "_" + crispasr_sep_lower(source_name) + "." + ext;
}

// Whether `source_name` is selected by a comma-separated `--stems` value.
// Empty (or "all") selects every stem. Matching is case-insensitive and
// tolerant of surrounding whitespace.
inline bool crispasr_stem_selected(const std::string& selection_csv, const std::string& source_name) {
    std::string sel = crispasr_sep_lower(selection_csv);
    // strip whitespace
    sel.erase(std::remove_if(sel.begin(), sel.end(), [](unsigned char c) { return std::isspace(c); }), sel.end());
    if (sel.empty() || sel == "all")
        return true;
    const std::string want = crispasr_sep_lower(source_name);
    size_t pos = 0;
    while (pos <= sel.size()) {
        const size_t comma = sel.find(',', pos);
        const std::string tok = sel.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (!tok.empty() && tok == want)
            return true;
        if (comma == std::string::npos)
            break;
        pos = comma + 1;
    }
    return false;
}

// Serialize stem `s` of a result to a WAV blob (interleaved int16 PCM, no
// AI-provenance tag — see crispasr_make_wav_int16_interleaved). Returns empty
// on an out-of-range index.
inline std::string crispasr_stem_to_wav(const crispasr_separation_view& v, int s) {
    if (s < 0 || s >= v.n_sources || !v.sources || !v.sources[s])
        return {};
    return crispasr_make_wav_int16_interleaved(v.sources[s], v.n_frames, v.n_channels, v.sample_rate);
}
