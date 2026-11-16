// crisp_punc.h — shared punctuation restoration library.
//
// Two backends:
//   FireRedPunc — BERT Chinese+multilingual, 5 classes
//   PCS — XLM-R, punctuation + capitalization + segmentation
//
// Used by both CrispASR and CrispEmbed for post-OCR/ASR text cleanup.

#ifndef CRISP_PUNC_H
#define CRISP_PUNC_H

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// FireRedPunc
// ---------------------------------------------------------------------------

struct fireredpunc_context;

// Load a FireRedPunc GGUF model.
struct fireredpunc_context* fireredpunc_init(const char* model_path);

// Add punctuation to unpunctuated text. Returns newly allocated string (caller frees).
char* fireredpunc_process(struct fireredpunc_context* ctx, const char* text);

// Free context.
void fireredpunc_free(struct fireredpunc_context* ctx);

// Parity hook: the token ids this engine feeds the model for `text`. Its
// tokenizer is private to fireredpunc.cpp (a second WordPiece implementation,
// separate from the shared one), so this is how a parity harness compares it
// against HuggingFace's BertTokenizer on the same vocab. The pointer is owned
// by the context and valid until the next call.
//
// Declared here as well as in CrispEmbed's fireredpunc.h because the two are
// the same implementation and the harness (CrispEmbed's firered-punct-ab) has
// to link against whichever copy the build selected.
const int* fireredpunc_debug_token_ids(struct fireredpunc_context* ctx, const char* text, int* out_n);

// ---------------------------------------------------------------------------
// PCS (Punctuation + Capitalization + Segmentation)
// ---------------------------------------------------------------------------

struct pcs_context;

// Load a PCS GGUF model.
struct pcs_context* pcs_init(const char* model_path);

// Apply punctuation, truecasing, and sentence boundary detection.
// Returns newly allocated string (caller frees).
char* pcs_process(struct pcs_context* ctx, const char* text);

// Free context.
void pcs_free(struct pcs_context* ctx);

#ifdef __cplusplus
}
#endif

#endif // CRISP_PUNC_H
