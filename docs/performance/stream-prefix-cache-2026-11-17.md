# 流式识别分阶段统计与固定提示 KV 复用

日期：2026-11-17。

本次优化针对每个新分段重复计算固定提示的问题。原先每个识别流单独保存固定提示 KV，
流关闭后丢弃；首次预填充还分成固定提示和剩余输入两次前向执行。
现在由模型会话保存固定提示缓存，跨分段、取消后重录及下一次录音复用。
缓存失效时通过一次完整的因果预填充同时生成固定提示 KV，省去单独的提示前向执行。
音频编码结果、音频位置的 KV 和识别文字都没有跨段复用。

## 分阶段结果

新增 `CRISPASR_QWEN3_STREAM_PROFILE=1` 诊断开关，每次实际生成记录一条
`QWEN3_STREAM_PERF`，包括无文字变化和失败的调用：

- `mel_ms`、`encoder_ms`、`prepare_ms`、`prefill_ms`、`generate_ms`：互不重叠的阶段。
- `audio_ms`、`audio_frames`、`prompt_tokens`、`generated_tokens`：输入规模与生成数量。
- `llm_graph_ms`、`llm_alloc_ms`、`llm_compute_ms`、`llm_transfer_ms`：嵌套于 LLM 阶段的明细，不能再次累加到总耗时。
- `prefix_reused`、`success`：本次是否复用固定提示，以及是否成功完成。

初始回放中，约 3 秒音频的单次识别耗时约 1.1 秒，编码器约 350～450ms，
预填充约 400ms，文字生成约 300ms；提示准备不到 1ms，LLM 构图与分配合计约 10～15ms。
因此先减少固定提示的重复推理，保留现有计算图和缓冲区的生命周期保护。

服务同时记录实际音频包入队等待：`queue_wait_ms` 是当前包最早样本的等待时间，
`queue_max_wait_ms` 是整次录音取包时观测到的最大等待；它们不包含取包后执行推理的时间。
时间戳随音频转存临时文件，长积压不会额外产生无界的内存元数据队列。

## A/B 方法与结果

机器：Intel Core i7-10700，CPU 推理，2 个线程，RelWithDebInfo。
模型：Qwen3-ASR-0.6B Q4_K。语料：`ko-369.wav`，使用 Korean 语言提示。

同一二进制创建两个独立模型上下文，分别预热，避免 A/B 相互改写 KV。
基线在每个独立分段重建固定提示缓存，并使用原先的分两次预填充；优化分支跨段保存缓存。
每个长度测量 3 次，按轮次交替 A/B 顺序。模型加载、预热及文件哈希计算不计入测量。
原始分阶段数据、模型和 WAV 的 SHA-256 见同目录 JSON。

| 音频长度 | 基线中位耗时 | 优化中位耗时 | 变化 |
|---|---:|---:|---:|
| 500ms | 418.6ms | 314.3ms | -24.9% |
| 1500ms | 757.1ms | 652.6ms | -13.8% |
| 3000ms | 1071.4ms | 966.3ms | -9.8% |

9 组配对调用的总耗时中位数从 757.1ms 降至 652.6ms，
下降 13.8%。每组识别文字和生成 token 数完全相同。
这是混合长度样本的中位耗时，不是端到端上屏延迟，也不代表中文 CER 评估。
固定提示的复用主要降低每段首次识别开销；段内更新原本就支持固定提示复用。

另以模拟采集进程按真实节奏送入 6 秒音频：优化后首次非空预览为 883ms，
松开前 5 次预览，松开后收尾 1280ms。该单次回放受调度与末次解码位置影响，
不据此声称收尾时间一定改善。一次性灌入 12 秒音频的积压测试和 RNNoise 路径均通过。

## 缓存正确性

固定提示只包含第一个音频占位符之前的因果 KV。模型运行时维护 KV 版本，
存储扩容、重置、前向计算写入都会更新版本。缓存仅在上下文、容量、提示 token 和版本均匹配时复用；
每次成功生成后刷新缓存版本。失败的前向即使已经部分写入 KV，也不会沿用旧版本。
不复制 KV 缓冲区，也不复用音频或用户转写对应的位置。

测试覆盖：失败统计清零、阶段计时汇总、缓存重置、容量扩展、其他推理插入，
以及流关闭重开、输入长度变化、Korean/Chinese 提示切换后的旧新路径输出一致性。
队列测试覆盖内存、转存、跨记录读取、部分读取、清空后的时间戳和样本顺序。
完整 CTest：38/38 通过。以上真实模型用例另外通过环境变量启用。

## 复现

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j4
./build/tests/stream_profile_benchmark \
  --model /path/to/qwen3-asr-0.6b-q4_k.gguf \
  --wav /path/to/speech.wav --language Korean \
  --threads 2 --repeats 3 --output /tmp/stream-profile.json
```

WAV 至少 3 秒；`--language` 使用模型的语言名称。输出 JSON 保存全部配对统计和转写哈希。
若推理失败、文字或 token 数不同、缓存复用状态不符合预期，工具返回非零，不发布成功报告。
输出仅比较基线与优化的一致性，不包含参考转写和 CER，也不是发布性能门禁证据。

原生流诊断可运行：

```bash
CRISPASR_QWEN3_STREAM_PROFILE=1 \
ECHOFLOW_QWEN3_TEST_MODEL=/path/to/qwen3-asr-0.6b-q4_k.gguf \
  ./build/tests/test_live_stream_transcriber pipelineHandlesBacklogWhenModelAvailable
```

模型用例还需本地 `third_party/crispasr/samples/ko-369.wav` 测试夹具。
默认不要求安装模型、采集设备或 Fcitx。线上服务的诊断开关需在进程启动前设置，
不会改变已经运行进程的环境。

排查时可在新流打开前设置以下环境变量，使用之前的执行方式进行对照：

```bash
CRISPASR_QWEN3_STREAM_REUSE_PREFIX=0
CRISPASR_QWEN3_STREAM_SPLIT_PREFILL=1
```

默认启用跨段固定提示复用和一次完整预填充。当前仍使用活动段累计重编码，
后续大幅优化需要针对编码器和音频预填充计算本身；不能直接冻结双向注意力的历史音频输出。
