// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef ECHOFLOW_CRISP_SESSION_H
#define ECHOFLOW_CRISP_SESSION_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct crispasr_session;
struct crispasr_session_result;
struct crispasr_stream;

namespace echoflow {

class CrispSession {
public:
    CrispSession(const std::string& modelPath, const std::string& backend,
                 int nThreads);
    ~CrispSession();

    CrispSession(const CrispSession&) = delete;
    CrispSession& operator=(const CrispSession&) = delete;

    bool isLoaded() const;
    void setLanguage(const std::string& lang);
    void setMaxNewTokens(int n);
    std::string transcribe(const float* pcm, int nSamples);

    int streamKind() const;
    bool openStream(int stepMs = 500, int lengthMs = 2000, int keepMs = 800);
    void setStreamLiveDecode(bool enabled);
    void setStreamFullSegmentDecode(bool enabled);
    bool feedStream(const float* pcm, int nSamples);
    std::int64_t streamInferenceCount() const;
    std::string streamText() const;
    std::string flushStream();
    std::string refreshStream();
    void closeStream();

    static std::vector<float> readWavF32(const std::string& path);

private:
    void updateStreamText();

    struct Result {
        ~Result();
        crispasr_session_result* r = nullptr;
        int nSegments = 0;
        std::string text() const;
    };
    Result transcribeResult(const float* pcm, int nSamples);

    crispasr_session* session_ = nullptr;
    crispasr_stream* stream_ = nullptr;
    int nThreads_ = 1;
    std::string language_;
    std::string streamText_;
};

}  // namespace echoflow

#endif
