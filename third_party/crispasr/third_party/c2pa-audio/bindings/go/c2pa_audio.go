// Package crispasrc2pa is a cgo binding for the standalone C2PA signer/verifier.
//
// It links the static library libc2pa_audio.a (build it with CMake first).
// No c2pa-rs. Signs and verifies C2PA Content Credentials in WAV audio.
package crispasrc2pa

/*
#cgo CFLAGS: -I${SRCDIR}/../../include
#cgo darwin LDFLAGS: -L${SRCDIR}/../../build -lc2pa_audio -lc++
#cgo linux LDFLAGS: -L${SRCDIR}/../../build -lc2pa_audio -lstdc++ -lm
#include <stdlib.h>
#include "c2pa_audio.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// VerifyResult reports which checks passed.
type VerifyResult struct {
	SignatureValid  bool
	DataHashValid   bool
	AssertionsValid bool
	Valid           bool
}

// Version returns the library version string.
func Version() string {
	return C.GoString(C.c2pa_audio_version())
}

// Sign signs data with a C2PA manifest. mime is "audio/wav" or "audio/mpeg".
// Pass empty certPem/keyPem to use the bundled self-signed default cert.
func Sign(data []byte, mime, certPem, keyPem string) ([]byte, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("empty input")
	}
	if mime == "" {
		mime = "audio/wav"
	}
	cMime := C.CString(mime)
	defer C.free(unsafe.Pointer(cMime))
	var out *C.uchar
	var outLen C.size_t
	var cCert, cKey *C.char
	if certPem != "" {
		cCert = C.CString(certPem)
		defer C.free(unsafe.Pointer(cCert))
	}
	if keyPem != "" {
		cKey = C.CString(keyPem)
		defer C.free(unsafe.Pointer(cKey))
	}
	rc := C.c2pa_audio_sign(
		(*C.uchar)(unsafe.Pointer(&data[0])), C.size_t(len(data)),
		cMime, cCert, cKey, &out, &outLen)
	if rc != 0 {
		return nil, fmt.Errorf("c2pa_audio_sign failed (rc=%d)", int(rc))
	}
	defer C.c2pa_audio_free(out)
	return C.GoBytes(unsafe.Pointer(out), C.int(outLen)), nil
}

// SignWav / SignMp3 are convenience wrappers over Sign.
func SignWav(wav []byte, certPem, keyPem string) ([]byte, error) {
	return Sign(wav, "audio/wav", certPem, keyPem)
}
func SignMp3(mp3 []byte, certPem, keyPem string) ([]byte, error) {
	return Sign(mp3, "audio/mpeg", certPem, keyPem)
}
func SignM4a(m4a []byte, certPem, keyPem string) ([]byte, error) {
	return Sign(m4a, "audio/mp4", certPem, keyPem)
}
func SignFlac(flac []byte, certPem, keyPem string) ([]byte, error) {
	return Sign(flac, "audio/flac", certPem, keyPem)
}

// Verify verifies a signed audio file (WAV or MP3, auto-detected).
func Verify(data []byte) VerifyResult {
	if len(data) == 0 {
		return VerifyResult{}
	}
	f := int(C.c2pa_audio_verify((*C.uchar)(unsafe.Pointer(&data[0])), C.size_t(len(data))))
	return VerifyResult{
		SignatureValid:  f&0x1 != 0,
		DataHashValid:   f&0x2 != 0,
		AssertionsValid: f&0x4 != 0,
		Valid:           f&0x8 != 0,
	}
}
