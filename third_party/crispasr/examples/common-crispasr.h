#pragma once

#include <string>
#include <vector>
#include <cstdint>

// Read WAV audio file and store the PCM data into pcmf32.
// fname can be a buffer of WAV data instead of a filename.
// target_rate: sample rate to decode to (0 = CRISPASR_SAMPLE_RATE = 16 kHz).
// If stereo flag is set and the audio has 2 channels, the pcmf32s will contain 2 channel PCM.
bool read_audio_data(const std::string& fname, std::vector<float>& pcmf32, std::vector<std::vector<float>>& pcmf32s,
                     bool stereo, int target_rate = 0);

// convert timestamp to string, 6000 -> 01:00.000
std::string to_timestamp(int64_t t, bool comma = false);

// given a timestamp get the sample
int timestamp_to_sample(int64_t t, int n_samples, int whisper_sample_rate);

// write text to file, and call system("command voice_id file")
bool speak_with_file(const std::string& command, const std::string& text, const std::string& path, int voice_id);
