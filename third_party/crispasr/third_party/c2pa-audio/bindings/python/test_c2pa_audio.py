import os, sys, struct, math
sys.path.insert(0, os.path.dirname(__file__))
from c2pa_audio import C2paAudio

def make_wav(n=4800, sr=24000):
    data=b''.join(struct.pack('<h',int(3000*math.sin(2*math.pi*220*i/sr))) for i in range(n))
    return b'RIFF'+struct.pack('<I',36+len(data))+b'WAVE'+b'fmt '+struct.pack('<IHHIIHH',16,1,1,sr,sr*2,2,16)+b'data'+struct.pack('<I',len(data))+data

c=C2paAudio()
print("version:", c.version)
wav=make_wav()
signed=c.sign_wav(wav)
assert len(signed)>len(wav), "sign failed"
r=c.verify(signed)
assert r.valid and r.signature_valid and r.data_hash_valid and r.assertions_valid, f"round-trip: {r}"
print("ok: round-trip", r)
t=bytearray(signed); t[46]^=0xff
assert not c.verify(bytes(t)).valid, "tamper not detected"
print("ok: tamper rejected")
ref=os.path.join(os.path.dirname(__file__),'..','..','test','assets','reference-c2pa-rs.wav')
if os.path.exists(ref):
    rr=c.verify(open(ref,'rb').read())
    assert rr.valid, f"c2pa-rs ref: {rr}"
    print("ok: c2pa-rs reference vector VALID")
import glob
mp3s=[p for p in [os.path.join(os.path.dirname(__file__),"..","..","test","assets","sample.mp3")] if os.path.exists(p)]
if mp3s:
    sm=c.sign(open(mp3s[0],"rb").read(),"audio/mpeg")
    assert c.verify(sm).valid, "MP3 round-trip"
    print("ok: MP3 round-trip VALID")
rm=os.path.join(os.path.dirname(__file__),"..","..","test","assets","reference-c2pa-rs.mp3")
if os.path.exists(rm):
    assert c.verify(open(rm,"rb").read()).valid, "c2pa-rs MP3 ref"
    print("ok: c2pa-rs MP3 reference VALID")
print("PASSED")
