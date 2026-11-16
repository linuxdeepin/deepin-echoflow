#!/usr/bin/env bash
# generate-c2pa-cert.sh — Generate a self-signed X.509 certificate for
# C2PA (Content Credentials) signing of CrispASR TTS output.
#
# The certificate is valid for 10 years and uses P-256 (EC). C2PA
# verifiers will show "unverified signer" for self-signed certs, but
# the manifest is still valid and machine-readable — sufficient for
# EU AI Act Article 50 compliance.
#
# For a trusted signer identity (green checkmark in Adobe Content
# Authenticity), obtain a code-signing certificate from a CA (Sectigo,
# DigiCert, GlobalSign — ~$70-300/year).
#
# Usage:
#   ./scripts/generate-c2pa-cert.sh [output-dir]
#
# Creates:
#   <output-dir>/crispasr-c2pa.crt  — PEM certificate
#   <output-dir>/crispasr-c2pa.key  — PEM private key (unencrypted)

set -euo pipefail

OUT_DIR="${1:-.}"
CERT="${OUT_DIR}/crispasr-c2pa.crt"
KEY="${OUT_DIR}/crispasr-c2pa.key"

if [ -f "$CERT" ] && [ -f "$KEY" ]; then
    echo "Certificate already exists: $CERT"
    echo "  To regenerate, delete the existing files first."
    exit 0
fi

echo "Generating self-signed C2PA certificate..."
# C2PA (and c2pa-rs) validate the signing cert against the spec's requirements:
# a leaf cert (CA:FALSE) with critical keyUsage=digitalSignature, a non-empty
# extendedKeyUsage (emailProtection), and Subject/Authority Key Identifiers.
# A bare `openssl req -x509` without these is rejected at sign time with
# "the certificate is invalid" — so pass them explicitly.
EXT_CNF="$(mktemp -t crispasr-c2pa-ext.XXXXXX)"
trap 'rm -f "$EXT_CNF"' EXIT
cat > "$EXT_CNF" <<'CNF'
[req]
distinguished_name = dn
x509_extensions = v3
prompt = no
[dn]
CN = CrispASR TTS
O = Self-Signed C2PA
[v3]
basicConstraints = critical, CA:FALSE
keyUsage = critical, digitalSignature
extendedKeyUsage = critical, emailProtection
subjectKeyIdentifier = hash
authorityKeyIdentifier = keyid:always
CNF
openssl req -x509 \
    -newkey ec \
    -pkeyopt ec_paramgen_curve:P-256 \
    -keyout "$KEY" \
    -out "$CERT" \
    -days 3650 \
    -nodes \
    -config "$EXT_CNF"

echo ""
echo "Certificate: $CERT"
echo "Private key: $KEY"
echo ""
echo "Use with CrispASR:"
echo "  crispasr --tts \"hello\" --c2pa-cert $CERT --c2pa-key $KEY"
echo ""
echo "Or in server mode:"
echo "  crispasr --server --c2pa-cert $CERT --c2pa-key $KEY"
