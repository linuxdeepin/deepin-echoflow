// crispasr_c2pa_default_cert.h — GENERATED. Do not edit by hand.
//
// A fixed, self-signed P-256/ES256 certificate + key baked into the binary so
// C2PA signing is on by default on EVERY platform, including the WASM browser
// sandbox and mobile where no openssl/subprocess is available to provision a
// per-install cert. The private key is intentionally PUBLIC (it ships in the
// artifact): this only marks content as AI-generated in a machine-readable way
// (self-signed → C2PA verifiers show "unverified signer"). It asserts nothing
// about identity and is NOT a trust anchor. For a TRUSTED signer identity, pass
// your own CA-issued cert via --c2pa-cert / --c2pa-key.
//
// Regenerate: openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:P-256
//   -nodes -days 3650 -keyout assets/c2pa/crispasr-default-c2pa.key
//   -out assets/c2pa/crispasr-default-c2pa.crt -config <ext.cnf with KU/EKU/SKI/AKI>
// then re-run scripts/gen-default-cert-header.sh.
#pragma once

inline const char* crispasr_c2pa_default_cert_pem() {
    return R"C2PA(-----BEGIN CERTIFICATE-----
MIIB/TCCAaSgAwIBAgIUMl0G7RMhbZrc3Aqvd+4XSk5c5YcwCgYIKoZIzj0EAwIw
QjEtMCsGA1UEAwwkQ3Jpc3BBU1IgKEFJLWdlbmVyYXRlZCwgc2VsZi1zaWduZWQp
MREwDwYDVQQKDAhDcmlzcEFTUjAeFw0yNjA3MTUxNTAxMzZaFw0zNjA3MTIxNTAx
MzZaMEIxLTArBgNVBAMMJENyaXNwQVNSIChBSS1nZW5lcmF0ZWQsIHNlbGYtc2ln
bmVkKTERMA8GA1UECgwIQ3Jpc3BBU1IwWTATBgcqhkjOPQIBBggqhkjOPQMBBwNC
AASg+eoWgUhN9yiih4LZ5tPnmUPEhNntVMT2lecs+4ifMxH8Ijsqa/SafmqAQ8Dj
nEnBKJv05YigE3QG/86KG8lxo3gwdjAMBgNVHRMBAf8EAjAAMA4GA1UdDwEB/wQE
AwIHgDAWBgNVHSUBAf8EDDAKBggrBgEFBQcDBDAdBgNVHQ4EFgQUROiy0IxEWq+K
dRMiX4Q191Hf2fowHwYDVR0jBBgwFoAUROiy0IxEWq+KdRMiX4Q191Hf2fowCgYI
KoZIzj0EAwIDRwAwRAIgFJ3cScgb91EYCeXBAkc6vWuAePi3VpTzaskL+8GSawAC
IBpjspvSj9n7ndRcGZHiWUG7tNyVFTmf9Hj4Ym/Gop+w
-----END CERTIFICATE-----
)C2PA";
}
inline const char* crispasr_c2pa_default_key_pem() {
    return R"C2PA(-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQg/9P4J5n66VxxuGET
+yu5U+0Z7zKRqZeklaMRurqtwrmhRANCAASg+eoWgUhN9yiih4LZ5tPnmUPEhNnt
VMT2lecs+4ifMxH8Ijsqa/SafmqAQ8DjnEnBKJv05YigE3QG/86KG8lx
-----END PRIVATE KEY-----
)C2PA";
}
