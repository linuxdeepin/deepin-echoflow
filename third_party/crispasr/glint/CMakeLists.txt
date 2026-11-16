# glint — our clean-room MP3 (MPEG-1/2/2.5 Layer III) + AAC-LC (ADTS,
# experimental phase 1) encoder, MIT licensed. In-tree copy of the
# sibling glint repository (encoder core only; the upstream CLI, tests,
# bindings and packaging stay there).
#
# Consumed by the TTS output paths: crispasr-server
# response_format=mp3|aac and crispasr-cli --tts-output *.mp3|*.aac
# (via examples/cli/crispasr_{mp3,aac}_writer.h).
#
# Built with CrispASR's global flags — no -march=native / -ffast-math from
# upstream's CMake. SIMD kernels (AVX/SSE2/NEON) are compile-time guarded
# and runtime dispatched, so default flags are safe on every target.

add_library(glint STATIC
    src/subband.cpp
    src/resample.cpp
    src/mdct.cpp
    src/quantize.cpp
    src/huffman.cpp
    src/reservoir.cpp
    src/bitstream.cpp
    src/psycho.cpp
    src/encoder.cpp
    src/aac_mdct.cpp
    src/aac_coder.cpp
    src/aac_psy.cpp
    src/aac_tns.cpp
    src/aac_encoder.cpp
    src/opus_ec.cpp
    src/opus_laplace.cpp
    src/opus_cwrs.cpp
    src/opus_celt_energy.cpp
    src/opus_celt_rate.cpp
    src/opus_celt_bands.cpp
    src/opus_celt_decoder.cpp
    src/opus_decoder.cpp
    src/opus_ms_decoder.cpp
    src/opus_c_api.cpp
    src/mp3_decoder.cpp
    src/aac_decoder.cpp
    src/flac_decoder.cpp
    src/decoder_c_api.cpp
    src/decode_audio_c_api.cpp
    src/wav_io.cpp
    src/wav_c_api.cpp
    src/encode_audio_c_api.cpp
    src/opus_mdct.cpp
    src/opus_ogg.cpp
    src/vorbis_decoder.cpp
    src/vorbis_c_api.cpp
    src/opus_celt_enc_energy.cpp
    src/opus_celt_enc_vq.cpp
    src/opus_celt_enc_bands.cpp
    src/opus_celt_encoder.cpp
    src/opus_analysis.cpp
    src/opus_celt_pitch.cpp
    src/opus_silk_excitation.cpp
    src/opus_silk_indices.cpp
    src/opus_silk_nlsf.cpp
    src/opus_silk_plc.cpp
    src/opus_silk_frame.cpp
    src/opus_silk_stereo.cpp
    src/opus_silk_resampler.cpp
    src/opus_silk_decoder.cpp
)

target_include_directories(glint
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

set_target_properties(glint PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)

# quantize.cpp's optional scale-factor search pool uses std::thread
find_package(Threads REQUIRED)
target_link_libraries(glint PRIVATE Threads::Threads)
