#!/bin/bash

cp -rpv ../llama.cpp/include/llama.h ./examples/talk-llama/llama.h

cp -rpv ../llama.cpp/src/llama*.cpp       ./examples/talk-llama/
cp -rpv ../llama.cpp/src/llama*.h         ./examples/talk-llama/
cp -rpv ../llama.cpp/src/models/*         ./examples/talk-llama/models/
cp -rpv ../llama.cpp/src/unicode.h        ./examples/talk-llama/unicode.h
cp -rpv ../llama.cpp/src/unicode.cpp      ./examples/talk-llama/unicode.cpp
cp -rpv ../llama.cpp/src/unicode-data.h   ./examples/talk-llama/unicode-data.h
cp -rpv ../llama.cpp/src/unicode-data.cpp ./examples/talk-llama/unicode-data.cpp

# Record which upstream revision this sync came from, the way
# scripts/sync-ggml.last does for ggml. Without a marker the vendored
# vintage can only be recovered by comparing every file in
# examples/talk-llama/ against upstream history, which stops working the
# moment anything reformats the copy.
git -C ../llama.cpp rev-parse HEAD > ./scripts/sync-llama.last
