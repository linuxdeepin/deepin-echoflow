// speaker_db.cpp — see speaker_db.h for format and API description.

#include "speaker_db.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#ifdef _WIN32
#include <io.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

static const char kMagic[4] = {'S', 'P', 'K', 'R'};
static const uint32_t kVersion = 2;

struct speaker_profile {
    std::string name;
    std::vector<float> embedding;
    bool consent_attested = false; // v1 files carry no record
    uint64_t enroll_time = 0;      // unix time; 0 for v1 files
};

struct speaker_db {
    std::string dir_path;
    std::vector<speaker_profile> speakers;
    // Set by speaker_db_retain(): the claimed closed roster. enroll_into()
    // consults it so same-handle enrollment can never widen the roster the
    // caller attested to (#266).
    std::vector<std::string> retained_roster;
    bool roster_applied = false;
};

// Read a .spkr file (v1 or v2). Returns false if the file is invalid.
static bool read_spkr_file(const char* path, speaker_profile& out) {
    FILE* f = fopen(path, "rb");
    if (!f)
        return false;

    char magic[4];
    uint32_t version = 0, dim = 0;
    bool ok = fread(magic, 1, 4, f) == 4 && memcmp(magic, kMagic, 4) == 0 && fread(&version, 4, 1, f) == 1 &&
              (version == 1 || version == kVersion) && fread(&dim, 4, 1, f) == 1 && dim > 0 && dim <= 4096;

    if (ok) {
        out.embedding.resize(dim);
        ok = fread(out.embedding.data(), sizeof(float), dim, f) == dim;
    }
    if (ok && version >= 2) {
        uint8_t consent = 0;
        uint64_t when = 0;
        ok = fread(&consent, 1, 1, f) == 1 && fread(&when, 8, 1, f) == 1;
        out.consent_attested = consent != 0;
        out.enroll_time = when;
    } else if (ok) {
        // Legacy v1 profile: no consent record was stored at enrollment.
        fprintf(stderr,
                "speaker_db: '%s' is a legacy v1 profile without a consent record; "
                "re-enroll to attach the consent attestation\n",
                path);
    }
    fclose(f);
    return ok;
}

extern "C" struct speaker_db* speaker_db_load(const char* dir_path) {
    if (!dir_path)
        return nullptr;

    auto* db = new speaker_db();
    db->dir_path = dir_path;

#ifdef _WIN32
    std::string pattern = std::string(dir_path) + "\\*.spkr";
    WIN32_FIND_DATAA fd;
    HANDLE hFind = FindFirstFileA(pattern.c_str(), &fd);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            std::string fname = fd.cFileName;
            std::string fullpath = std::string(dir_path) + "\\" + fname;
            speaker_profile sp;
            if (read_spkr_file(fullpath.c_str(), sp)) {
                sp.name = fname.substr(0, fname.size() - 5); // strip ".spkr"
                db->speakers.push_back(std::move(sp));
            }
        } while (FindNextFileA(hFind, &fd));
        FindClose(hFind);
    }
#else
    DIR* d = opendir(dir_path);
    if (!d) {
        // Directory doesn't exist — return empty db (not an error, user may create it later)
        return db;
    }
    struct dirent* ent;
    while ((ent = readdir(d)) != nullptr) {
        std::string fname = ent->d_name;
        if (fname.size() <= 5 || fname.substr(fname.size() - 5) != ".spkr")
            continue;
        std::string fullpath = std::string(dir_path) + "/" + fname;
        speaker_profile sp;
        if (read_spkr_file(fullpath.c_str(), sp)) {
            sp.name = fname.substr(0, fname.size() - 5);
            db->speakers.push_back(std::move(sp));
        }
    }
    closedir(d);
#endif

    if (!db->speakers.empty())
        fprintf(stderr, "speaker_db: loaded %zu speakers from %s\n", db->speakers.size(), dir_path);

    return db;
}

extern "C" void speaker_db_free(struct speaker_db* db) {
    delete db;
}

extern "C" int speaker_db_count(const struct speaker_db* db) {
    return db ? (int)db->speakers.size() : 0;
}

extern "C" const char* speaker_db_name(const struct speaker_db* db, int idx) {
    if (!db || idx < 0 || idx >= (int)db->speakers.size())
        return nullptr;
    return db->speakers[idx].name.c_str();
}

extern "C" int speaker_db_retain(struct speaker_db* db, const char* csv_names) {
    if (!db)
        return 0;
    if (!csv_names || !*csv_names) {
        // No roster claimed — retain nothing. An unclaimed db must never
        // silently fall back to an open 1:N search.
        db->speakers.clear();
        db->retained_roster.clear();
        db->roster_applied = true;
        return 0;
    }

    std::vector<std::string> claimed;
    std::string cur;
    for (const char* p = csv_names;; p++) {
        if (*p == ',' || *p == '\0') {
            size_t b = cur.find_first_not_of(" \t");
            size_t e = cur.find_last_not_of(" \t");
            if (b != std::string::npos)
                claimed.push_back(cur.substr(b, e - b + 1));
            cur.clear();
            if (*p == '\0')
                break;
        } else {
            cur += *p;
        }
    }

    db->retained_roster = claimed;
    db->roster_applied = true;

    std::vector<speaker_profile> kept;
    for (const auto& name : claimed) {
        bool found = false;
        for (auto& sp : db->speakers) {
            if (sp.name == name) {
                kept.push_back(sp);
                found = true;
                break;
            }
        }
        if (!found)
            fprintf(stderr, "speaker_db: claimed speaker '%s' has no enrolled profile in %s\n", name.c_str(),
                    db->dir_path.c_str());
    }
    db->speakers = std::move(kept);
    return (int)db->speakers.size();
}

extern "C" const char* speaker_db_match(const struct speaker_db* db, const float* embedding, int dim, float threshold,
                                        float* out_score) {
    if (!db || !embedding || dim <= 0 || db->speakers.empty())
        return nullptr;

    float best_score = -2.0f;
    int best_idx = -1;

    for (int i = 0; i < (int)db->speakers.size(); i++) {
        auto& sp = db->speakers[i];
        if ((int)sp.embedding.size() != dim)
            continue;

        // Cosine similarity (dot product for L2-normed vectors)
        float dot = 0.0f;
        for (int k = 0; k < dim; k++)
            dot += embedding[k] * sp.embedding[k];

        if (dot > best_score) {
            best_score = dot;
            best_idx = i;
        }
    }

    if (out_score)
        *out_score = best_score;

    if (best_idx >= 0 && best_score >= threshold)
        return db->speakers[best_idx].name.c_str();

    return nullptr;
}

extern "C" bool speaker_db_enroll(const char* dir_path, const char* name, const float* embedding, int dim,
                                  bool consent_attested) {
    if (!dir_path || !name || !embedding || dim <= 0)
        return false;
    if (!consent_attested) {
        fprintf(stderr, "speaker_db: enrollment refused: a voiceprint linked to a real name is biometric\n"
                        "  data (GDPR Art. 9); the caller must attest the enrolled person's explicit consent\n");
        return false;
    }

    // Ensure directory exists
#ifdef _WIN32
    CreateDirectoryA(dir_path, nullptr);
#else
    mkdir(dir_path, 0755);
#endif

    std::string path = std::string(dir_path) + "/" + std::string(name) + ".spkr";
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "speaker_db: cannot write %s\n", path.c_str());
        return false;
    }

    uint32_t udim = (uint32_t)dim;
    uint8_t consent = 1;
    uint64_t when = (uint64_t)time(nullptr);
    bool ok = fwrite(kMagic, 1, 4, f) == 4 && fwrite(&kVersion, 4, 1, f) == 1 && fwrite(&udim, 4, 1, f) == 1 &&
              fwrite(embedding, sizeof(float), dim, f) == (size_t)dim && fwrite(&consent, 1, 1, f) == 1 &&
              fwrite(&when, 8, 1, f) == 1;

    fclose(f);
    if (ok)
        fprintf(stderr, "speaker_db: enrolled '%s' → %s (%d-d, consent recorded)\n", name, path.c_str(), dim);
    return ok;
}

extern "C" bool speaker_db_enroll_into(struct speaker_db* db, const char* name, const float* embedding, int dim,
                                       bool consent_attested) {
    if (!db || !name || !embedding || dim <= 0)
        return false;
    if (!speaker_db_enroll(db->dir_path.c_str(), name, embedding, dim, consent_attested))
        return false;

    // The disk write is done; decide whether THIS handle may match the name.
    if (db->roster_applied) {
        bool on_roster = false;
        for (const auto& r : db->retained_roster) {
            if (r == name) {
                on_roster = true;
                break;
            }
        }
        if (!on_roster) {
            fprintf(stderr,
                    "speaker_db: '%s' enrolled on disk but NOT added to this handle — it is outside the\n"
                    "  retained roster; reopen with the name claimed to match against it (#266)\n",
                    name);
            return true;
        }
    }

    speaker_profile sp;
    sp.name = name;
    sp.embedding.assign(embedding, embedding + dim);
    sp.consent_attested = true;
    sp.enroll_time = (uint64_t)time(nullptr);
    for (auto& existing : db->speakers) {
        if (existing.name == sp.name) {
            existing = std::move(sp);
            return true;
        }
    }
    db->speakers.push_back(std::move(sp));
    return true;
}
