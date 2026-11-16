package crispasrc2pa

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func makeWav() []byte {
	sr, n := 24000, 4800
	var b []byte
	u32 := func(v int) { b = append(b, byte(v), byte(v>>8), byte(v>>16), byte(v>>24)) }
	u16 := func(v int) { b = append(b, byte(v), byte(v>>8)) }
	b = append(b, "RIFF"...); u32(36 + n*2); b = append(b, "WAVE"...)
	b = append(b, "fmt "...); u32(16); u16(1); u16(1); u32(sr); u32(sr * 2); u16(2); u16(16)
	b = append(b, "data"...); u32(n * 2)
	for i := 0; i < n; i++ {
		s := int(3000 * math.Sin(2*math.Pi*220*float64(i)/float64(sr)))
		u16(s & 0xffff)
	}
	return b
}

func TestRoundTrip(t *testing.T) {
	if Version() == "" { t.Fatal("empty version") }
	signed, err := SignWav(makeWav(), "", "")
	if err != nil { t.Fatal(err) }
	if len(signed) <= len(makeWav()) { t.Fatal("sign produced no manifest") }
	r := Verify(signed)
	if !r.Valid || !r.SignatureValid || !r.DataHashValid || !r.AssertionsValid {
		t.Fatalf("round-trip not valid: %+v", r)
	}
}

func TestTamper(t *testing.T) {
	signed, _ := SignWav(makeWav(), "", "")
	signed[46] ^= 0xff
	if Verify(signed).Valid { t.Fatal("tamper not detected") }
}

func TestReferenceVector(t *testing.T) {
	p := filepath.Join("..", "..", "test", "assets", "reference-c2pa-rs.wav")
	data, err := os.ReadFile(p)
	if err != nil { t.Skip("reference vector missing") }
	if !Verify(data).Valid { t.Fatal("c2pa-rs reference vector did not validate") }
}

func TestMp3(t *testing.T) {
	p := filepath.Join("..", "..", "test", "assets", "sample.mp3")
	data, err := os.ReadFile(p)
	if err != nil { t.Skip("sample.mp3 missing") }
	signed, err := SignMp3(data, "", "")
	if err != nil { t.Fatal(err) }
	if !Verify(signed).Valid { t.Fatal("MP3 round-trip not valid") }
	ref, err := os.ReadFile(filepath.Join("..", "..", "test", "assets", "reference-c2pa-rs.mp3"))
	if err == nil && !Verify(ref).Valid { t.Fatal("c2pa-rs MP3 ref did not validate") }
}
