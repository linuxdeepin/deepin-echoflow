// glint - Bit reservoir
// MIT License - Clean-room implementation

#ifndef GLINT_RESERVOIR_HPP
#define GLINT_RESERVOIR_HPP

#include <cstdint>
#include <cstring>
#include <algorithm>
#include "bitstream.hpp"

namespace glint {

// Bit reservoir as a continuous main-data byte stream with deferred frame
// emission.
//
// MP3 main data is a continuous byte stream, independent of the frame slots:
// each frame's slot carries a fixed number of main-data bytes (slot_md), but a
// frame's actual granule data may start up to `main_data_begin` bytes *before*
// its own slot — i.e. in the unused tail of earlier frames' slots. This lets a
// demanding frame borrow space that leaner neighbours left unused.
//
// Because a frame's data can spill into a *later* frame's slot, a slot can only
// be emitted once enough subsequent data has been produced to fill it. We
// therefore buffer pending frame headers and emit complete slots as the stream
// fills; the last frame(s) are released by flush().
//
// Invariant kept by the caller's bit budget: a frame produces at most
// slot_md + main_data_begin() bytes, so the stream's slot cursor never falls
// behind the produced data (main_data_begin stays >= 0).
//
// STATUS: this mechanism is correct and verified (0 decode backstep errors,
// unit tests pass). It is NOT yet wired to spend reservoir bits, because a
// naive "let every frame use the whole reservoir" policy regresses quality.
// See the long comment at the available_bits computation in encoder.cpp for
// the measurements and what a beneficial perceptual allocation policy needs.
class ReservoirStream {
public:
    void init(int resv_max_bytes) {
        resv_max_ = resv_max_bytes;     // 9-bit field => 511 (MPEG-1); 255 (MPEG-2)
        slot_start_ = 0;
        data_pos_ = 0;
        head_ = 0;
        stream_len_ = 0;
        n_pending_ = 0;
    }

    // Reservoir bytes available to the frame about to be encoded. This is the
    // value to write into the side-info main_data_begin field, and 8x this is
    // the extra bit budget the quantizer may use beyond one slot.
    int main_data_begin() const {
        long mdb = slot_start_ - data_pos_;
        if (mdb < 0) mdb = 0;
        if (mdb > resv_max_) mdb = resv_max_;
        return static_cast<int>(mdb);
    }

    // Append one encoded frame. header_si is header(4) + side_info bytes;
    // md is this frame's byte-aligned main data. slot_md is the number of
    // main-data bytes this frame's slot carries (frame_size - header_si_len).
    // Writes any now-complete frames into out_buf and returns bytes written.
    int add_frame(const uint8_t* header_si, int hs_len,
                  const uint8_t* md, int md_len, int slot_md,
                  uint8_t* out_buf, int out_cap) {
        // Append this frame's main data to the stream.
        append(md, md_len);

        // Cap the reservoir: if this frame was lean enough that its data sits
        // more than resv_max_ bytes before the next slot, the gap can't be
        // expressed by the (bounded) main_data_begin field. Pad the stream with
        // stuffing bytes so the gap stays within range. The stuffing lands in a
        // slot tail that no frame's data references, so decoders skip it.
        long gap = (slot_start_ + slot_md) - data_pos_;
        if (gap > resv_max_) {
            int pad = static_cast<int>(gap - resv_max_);
            pad_zeros(pad);
        }

        // Queue this frame.
        Pending& p = pending_[n_pending_++];
        std::memcpy(p.hs, header_si, hs_len);
        p.hs_len = hs_len;
        p.slot_start = slot_start_;
        p.slot_md = slot_md;
        slot_start_ += slot_md;

        return drain(out_buf, out_cap, /*final=*/false);
    }

    // Release all buffered frames at end of stream, padding the final slots.
    int flush(uint8_t* out_buf, int out_cap) {
        // Ensure every queued slot can be filled.
        if (data_pos_ < slot_start_)
            pad_zeros(static_cast<int>(slot_start_ - data_pos_));
        return drain(out_buf, out_cap, /*final=*/true);
    }

private:
    void append(const uint8_t* p, int n) {
        std::memcpy(stream_ + stream_len_, p, n);
        stream_len_ += n;
        data_pos_ += n;
    }
    void pad_zeros(int n) {
        std::memset(stream_ + stream_len_, 0, n);
        stream_len_ += n;
        data_pos_ += n;
    }

    int drain(uint8_t* out_buf, int out_cap, bool final) {
        int written = 0;
        int i = 0;
        while (i < n_pending_) {
            Pending& p = pending_[i];
            long slot_end = p.slot_start + p.slot_md;
            if (!final && data_pos_ < slot_end) break;  // slot not yet full

            int hs = p.hs_len;
            (void)out_cap;  // buffer is sized by the caller for the worst case
            std::memcpy(out_buf + written, p.hs, hs);
            written += hs;
            // Slot main-data bytes start at p.slot_start in the stream; the
            // live stream window begins at head_.
            int off = static_cast<int>(p.slot_start - head_);
            std::memcpy(out_buf + written, stream_ + off, p.slot_md);
            written += p.slot_md;

            // Discard stream bytes now fully emitted.
            int new_head_off = static_cast<int>(slot_end - head_);
            std::memmove(stream_, stream_ + new_head_off, stream_len_ - new_head_off);
            stream_len_ -= new_head_off;
            head_ = slot_end;
            i++;
        }
        // Drop the emitted entries from the front of the queue.
        if (i > 0) {
            for (int k = i; k < n_pending_; k++) pending_[k - i] = pending_[k];
            n_pending_ -= i;
        }
        return written;
    }

    // Live stream window: stream_[0 .. stream_len_) corresponds to absolute
    // byte indices [head_ .. data_pos_).
    uint8_t stream_[kMainDataBufSize];
    long slot_start_;   // absolute slot cursor (sum of queued slot_md)
    long data_pos_;     // absolute end of produced main data (= head_ + stream_len_)
    long head_;         // absolute index of stream_[0]
    int stream_len_;
    int resv_max_;

    struct Pending {
        uint8_t hs[64];
        int hs_len;
        long slot_start;
        int slot_md;
    };
    Pending pending_[16];
    int n_pending_;
};

} // namespace glint

#endif // GLINT_RESERVOIR_HPP
