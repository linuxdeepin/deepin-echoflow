// speaker_db.h — simple file-based speaker profile database.
//
// Each speaker is stored as a file in a directory:
//   <db_path>/<name>.spkr
//
// File format (version 2):
//   4 bytes: magic "SPKR"
//   4 bytes: uint32 version (2)
//   4 bytes: uint32 embedding dimension (192)
//   dim * 4 bytes: float32 L2-normalized embedding
//   1 byte:  uint8 consent_attested (1 = enrollment consent affirmed)
//   8 bytes: uint64 enrollment unix time (consent audit trail)
// Version-1 files (no consent trailer) still load, with a stderr notice.
//
// Matching: cosine similarity (dot product for L2-normed vectors).
//
// Privacy posture (issue #266): profiles are biometric special-category
// data (GDPR Art. 9). Matching is intended ONLY against a closed roster
// of claimed, consenting participants — callers must narrow a loaded db
// to the claimed names via speaker_db_retain() before matching. There
// is deliberately no open-ended "identify anyone in the db" mode.

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct speaker_db;

// Load all speaker profiles from a directory. Returns NULL on failure
// (e.g. directory doesn't exist). An empty directory is valid (0 speakers).
struct speaker_db* speaker_db_load(const char* dir_path);

// Free all resources.
void speaker_db_free(struct speaker_db* db);

// Return the number of enrolled speakers.
int speaker_db_count(const struct speaker_db* db);

// Return the name of the idx-th profile (0-based), or NULL if out of
// range. Valid until the db is freed or speaker_db_retain() is called.
const char* speaker_db_name(const struct speaker_db* db, int idx);

// Narrow the db to a closed roster: keep only profiles whose name is in
// `csv_names` (comma-separated, surrounding whitespace trimmed). Warns on
// stderr for claimed names that have no enrolled profile. Returns the
// number of profiles retained. Matching after retain() is a claimed-
// participant confirmation, not an open 1:N search.
int speaker_db_retain(struct speaker_db* db, const char* csv_names);

// Match an embedding against the database.
// Returns the best-matching speaker name if cosine_sim >= threshold.
// Returns NULL if no match or database is empty.
// The returned string is valid until the db is freed.
// If `out_score` is non-NULL, writes the best cosine similarity score.
const char* speaker_db_match(const struct speaker_db* db, const float* embedding, int dim, float threshold,
                             float* out_score);

// Enroll a speaker: save an L2-normalized embedding to <dir_path>/<name>.spkr.
// If the file already exists, it is overwritten.
// `consent_attested` records that the caller affirmed the enrolled person's
// explicit consent (GDPR Art. 9); enrollment REFUSES when it is false.
// Returns true on success.
bool speaker_db_enroll(const char* dir_path, const char* name, const float* embedding, int dim, bool consent_attested);

// Enroll THROUGH an open handle: writes <dir>/<name>.spkr exactly like
// speaker_db_enroll() AND updates the handle's in-memory profiles, so a
// subsequent speaker_db_match() on the same handle can resolve the name —
// previously the handle never learned of the new profile and callers had
// to know an undocumented close-and-reopen contract.
//
// Closed-roster guarantee (#266) is preserved: when speaker_db_retain()
// has been applied, a name OUTSIDE the retained roster is still written
// to disk (available to future handles that claim it) but is NOT added
// to this handle's matchable set; a note is printed to stderr.
// Returns true when the on-disk write succeeded.
bool speaker_db_enroll_into(struct speaker_db* db, const char* name, const float* embedding, int dim,
                            bool consent_attested);

#ifdef __cplusplus
}
#endif
