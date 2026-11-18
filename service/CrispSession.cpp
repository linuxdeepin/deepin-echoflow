// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "CrispSession.h"

extern "C" {
#include "crispasr_session.h"
}

#include "log.h"

#include <samplerate.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace echoflow {

// ---------------------------------------------------------------------------
// WAV reader
// ---------------------------------------------------------------------------

namespace {
uint64_t readLittleEndian(const unsigned char* bytes, size_t count)
{
    uint64_t value = 0;
    for (size_t i = 0; i < count; ++i) value |= uint64_t(bytes[i]) << (8 * i);
    return value;
}
}

std::vector<float> CrispSession::readWavF32(const std::string& path)
{
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    const auto invalid = [&path] {
        log("invalid or unsupported WAV file: " + path);
        return std::vector<float>();
    };
    if (!in || in.tellg() < 12) return invalid();
    const auto fileSize = static_cast<uint64_t>(in.tellg());
    const auto readAt = [&in](uint64_t offset, unsigned char* bytes, size_t count) {
        in.clear();
        in.seekg(static_cast<std::streamoff>(offset));
        in.read(reinterpret_cast<char*>(bytes), static_cast<std::streamsize>(count));
        return static_cast<size_t>(in.gcount()) == count;
    };
    unsigned char header[12];
    if (!readAt(0, header, sizeof(header)) || std::memcmp(header, "RIFF", 4)
        || std::memcmp(header + 8, "WAVE", 4)) return invalid();
    unsigned format = 0, channels = 0, bits = 0, rate = 0, align = 0;
    uint64_t dataOffset = 0, dataBytes = 0;
    for (uint64_t offset = 12; offset + 8 <= fileSize;) {
        unsigned char chunk[8];
        if (!readAt(offset, chunk, sizeof(chunk))) return invalid();
        const auto count = readLittleEndian(chunk + 4, 4);
        if (count > fileSize - offset - 8) return invalid();
        if (std::memcmp(chunk, "fmt ", 4) == 0) {
            unsigned char fmt[40] = {};
            if (count < 16 || !readAt(offset + 8, fmt, static_cast<size_t>(std::min<uint64_t>(count, sizeof(fmt))))) return invalid();
            format = static_cast<unsigned>(readLittleEndian(fmt, 2));
            channels = static_cast<unsigned>(readLittleEndian(fmt + 2, 2));
            rate = static_cast<unsigned>(readLittleEndian(fmt + 4, 4));
            align = static_cast<unsigned>(readLittleEndian(fmt + 12, 2));
            bits = static_cast<unsigned>(readLittleEndian(fmt + 14, 2));
            if (format == 0xfffe) {
                const unsigned char guidTail[12] = {0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71};
                if (count < 40 || readLittleEndian(fmt + 16, 2) < 22
                    || std::memcmp(fmt + 28, guidTail, sizeof(guidTail))) return invalid();
                format = static_cast<unsigned>(readLittleEndian(fmt + 24, 4));
            }
        } else if (std::memcmp(chunk, "data", 4) == 0 && !dataOffset) {
            dataOffset = offset + 8;
            dataBytes = count;
        }
        offset += 8 + count + (count & 1); // RIFF chunks have even-byte padding.
    }
    if (!dataOffset || !dataBytes || channels == 0 || channels > 64 || rate < 1000 || rate > 384000
        || !((format == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32))
             || (format == 3 && (bits == 32 || bits == 64)))
        || align != channels * (bits / 8) || dataBytes % align) return invalid();
    const size_t sampleBytes = bits / 8;
    std::vector<float> pcm;
    pcm.reserve(static_cast<size_t>(dataBytes / align));
    const size_t capacity = (8192 / align) * align;
    std::vector<unsigned char> buffer(capacity);
    for (uint64_t offset = 0; offset < dataBytes;) {
        const size_t count = static_cast<size_t>(std::min<uint64_t>(capacity, dataBytes - offset));
        if (!readAt(dataOffset + offset, buffer.data(), count)) return invalid();
        for (size_t frame = 0; frame < count; frame += align) {
            double sum = 0;
            for (size_t channel = 0; channel < channels; ++channel) {
                const auto raw = readLittleEndian(buffer.data() + frame + channel * sampleBytes, sampleBytes);
                double value;
                if (format == 3 && bits == 32) {
                    const auto word = static_cast<uint32_t>(raw);
                    float sample;
                    std::memcpy(&sample, &word, sizeof(sample));
                    value = sample;
                } else if (format == 3) {
                    std::memcpy(&value, &raw, sizeof(value));
                } else if (bits == 8) {
                    value = (static_cast<int>(raw) - 128) / 128.0;
                } else {
                    const auto scale = int64_t(1) << (bits - 1);
                    const auto sample = raw >= static_cast<uint64_t>(scale)
                        ? static_cast<int64_t>(raw) - 2 * scale : static_cast<int64_t>(raw);
                    value = static_cast<double>(sample) / scale;
                }
                sum += std::isfinite(value) ? std::clamp(value, -1.0, 1.0) : 0;
            }
            pcm.push_back(static_cast<float>(sum / channels));
        }
        offset += count;
    }
    if (rate != 16000) {
        const double ratio = 16000.0 / rate;
        std::vector<float> converted(static_cast<size_t>(std::ceil(pcm.size() * ratio)) + 64);
        SRC_DATA data {};
        data.data_in = pcm.data();
        data.input_frames = static_cast<long>(pcm.size());
        data.data_out = converted.data();
        data.output_frames = static_cast<long>(converted.size());
        data.src_ratio = ratio;
        data.end_of_input = 1;
        const int error = src_simple(&data, SRC_SINC_FASTEST, 1);
        if (error || data.input_frames_used != data.input_frames) return invalid();
        converted.resize(static_cast<size_t>(data.output_frames_gen));
        pcm = std::move(converted);
    }
    return pcm;
}

// ---------------------------------------------------------------------------
// CrispSession
// ---------------------------------------------------------------------------

CrispSession::CrispSession(const std::string& modelPath, const std::string& backend,
                           int nThreads)
    : nThreads_(nThreads)
{
    session_ = crispasr_session_open_explicit(modelPath.c_str(), backend.c_str(),
                                               nThreads);
    if (!session_) {
        log("crispasr_session_open_explicit failed for: " + modelPath);
    }
}

CrispSession::~CrispSession()
{
    closeStream();
    if (session_) {
        crispasr_session_close(session_);
    }
}

bool CrispSession::isLoaded() const
{
    return session_ != nullptr;
}

void CrispSession::setLanguage(const std::string& lang)
{
    language_ = lang;
    if (session_) {
        crispasr_session_set_source_language(session_, lang.c_str());
    }
}

void CrispSession::setMaxNewTokens(int n)
{
    if (session_ && n > 0) {
        crispasr_session_set_max_new_tokens(session_, n);
    }
}

CrispSession::Result CrispSession::transcribeResult(const float* pcm, int nSamples)
{
    Result res;
    res.r = crispasr_session_transcribe(session_, pcm, nSamples);
    if (res.r) {
        res.nSegments = crispasr_session_result_n_segments(res.r);
    }
    return res;
}

std::string CrispSession::transcribe(const float* pcm, int nSamples)
{
    return transcribeResult(pcm, nSamples).text();
}

int CrispSession::streamKind() const
{
    return session_ ? crispasr_session_stream_kind(session_) : 0;
}

bool CrispSession::openStream(int stepMs, int lengthMs, int keepMs)
{
    closeStream();
    if (!session_ || streamKind() == 0) {
        return false;
    }
    stream_ = crispasr_session_stream_open(session_, nThreads_, stepMs, lengthMs,
        keepMs, language_.empty() ? nullptr : language_.c_str(), 0);
    if (!stream_) {
        return false;
    }
    streamText_.clear();
    return true;
}

void CrispSession::setStreamLiveDecode(bool enabled)
{
    if (stream_) crispasr_stream_set_live_decode(stream_, enabled ? 1 : 0);
}

void CrispSession::setStreamFullSegmentDecode(bool enabled)
{
    if (stream_) crispasr_stream_set_full_segment_decode(stream_, enabled ? 1 : 0);
}

std::int64_t CrispSession::streamInferenceCount() const
{
    return crispasr_stream_inference_count(stream_);
}

bool CrispSession::feedStream(const float* pcm, int nSamples)
{
    if (!stream_ || !pcm || nSamples <= 0) {
        return false;
    }
    const int rc = crispasr_stream_feed(stream_, pcm, nSamples);
    if (rc < 0) {
        throw std::runtime_error("crispasr stream feed failed: " + std::to_string(rc));
    }
    if (rc == 0) {
        return false;
    }

    const auto previous = streamText_;
    updateStreamText();
    return streamText_ != previous;
}

void CrispSession::updateStreamText()
{
    std::vector<char> buffer(4096, '\0');
    int size = crispasr_stream_get_text(stream_, buffer.data(), buffer.size(),
        nullptr, nullptr, nullptr);
    if (size < 0) throw std::runtime_error("crispasr stream text retrieval failed");
    if (static_cast<size_t>(size) >= buffer.size()) {
        buffer.resize(static_cast<size_t>(size) + 1, '\0');
        size = crispasr_stream_get_text(stream_, buffer.data(), buffer.size(),
            nullptr, nullptr, nullptr);
        if (size < 0) throw std::runtime_error("crispasr stream text retrieval failed");
    }
    // Empty is a valid retraction of the preceding hypothesis.
    streamText_.assign(buffer.data(), static_cast<size_t>(size));
}

std::string CrispSession::streamText() const
{
    return streamText_;
}

std::string CrispSession::flushStream()
{
    if (!stream_) {
        return streamText_;
    }
    const int rc = crispasr_stream_flush(stream_);
    if (rc < 0) {
        throw std::runtime_error("crispasr stream flush failed: " + std::to_string(rc));
    }
    updateStreamText();
    return streamText_;
}

std::string CrispSession::refreshStream()
{
    if (!stream_) return streamText_;
    const int rc = crispasr_stream_refresh(stream_);
    if (rc < 0) {
        throw std::runtime_error("crispasr stream refresh failed: " + std::to_string(rc));
    }
    updateStreamText();
    return streamText_;
}

void CrispSession::closeStream()
{
    if (stream_) {
        crispasr_stream_close(stream_);
        stream_ = nullptr;
    }
    streamText_.clear();
}

// ---------------------------------------------------------------------------
// CrispSession::Result
// ---------------------------------------------------------------------------

CrispSession::Result::~Result()
{
    if (r) {
        crispasr_session_result_free(r);
    }
}

std::string CrispSession::Result::text() const
{
    std::string out;
    for (int i = 0; i < nSegments; ++i) {
        const char* seg = crispasr_session_result_segment_text(r, i);
        if (seg) {
            if (i > 0 && !out.empty() && out.back() != '\n') {
                out += " ";
            }
            out += seg;
        }
    }
    return out;
}

}  // namespace echoflow
