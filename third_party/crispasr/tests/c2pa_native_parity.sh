#!/usr/bin/env bash
# c2pa_native_parity.sh — live parity test for the native C++ C2PA signer.
#
# Emits a signed WAV via the test binary ([emit] case), then validates it in the
# c2pa-rs reference reader through the c2pa-python package. Proves the from-
# scratch C++ ES256 / CBOR / JUMBF / COSE output is interoperable with the
# canonical implementation. Skips (exit 0) when python or the c2pa package is
# unavailable so it never blocks a build without the reference reader.
#
# Usage: c2pa_native_parity.sh <path-to-test_c2pa_native-binary>
set -u
BIN="${1:?usage: c2pa_native_parity.sh <test_c2pa_native binary>}"

# Locate a python with c2pa importable.
PY=""
for cand in "${C2PA_PY:-}" python3 python; do
    [ -z "$cand" ] && continue
    if "$cand" -c 'import c2pa' >/dev/null 2>&1; then PY="$cand"; break; fi
done
if [ -z "$PY" ]; then
    echo "SKIP: no python with the c2pa package (set C2PA_PY to enable parity)"
    exit 0
fi

WAV="$(mktemp -t crispasr-c2pa-native-XXXXXX).wav"
trap 'rm -f "$WAV"' EXIT

CRISPASR_C2PA_EMIT="$WAV" "$BIN" "[emit]" >/dev/null 2>&1 || { echo "FAIL: emit failed"; exit 1; }
[ -s "$WAV" ] || { echo "FAIL: no WAV emitted"; exit 1; }

"$PY" - "$WAV" <<'PY'
import sys, json
from c2pa import Reader
with open(sys.argv[1], "rb") as f:
    m = json.loads(Reader("audio/wav", f).json())
statuses = [v["code"] for v in m.get("validation_status", [])]
bad = [c for c in statuses if c != "signingCredential.untrusted"]
if bad:
    print("FAIL: unexpected validation status:", bad); sys.exit(1)
am = m["manifests"][m["active_manifest"]]
assert am["claim_generator_info"][0]["name"] == "CrispASR", "generator mismatch"
actions = [a for a in am["assertions"] if a["label"] == "c2pa.actions.v2"]
assert actions, "missing c2pa.actions.v2"
act = actions[0]["data"]["actions"][0]
assert act["action"] == "c2pa.created", "action mismatch"
assert act["digitalSourceType"].endswith("trainedAlgorithmicMedia"), "dst mismatch"
assert str(am.get("signature_info", {}).get("alg", "")).lower() == "es256", "alg mismatch"
print("PASS: native C++ C2PA validates in the c2pa-rs reference reader")
PY
