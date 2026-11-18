<p align="center">
  <img src="ui-host/icons/deepin-echoflow.svg" width="128" height="128" alt="deepin-echoflow logo">
</p>

<h1 align="center">deepin-echoflow</h1>

<p align="center">
  <strong>为 deepin 打造的离线语音输入法</strong>
</p>

<p align="center">
  <a href="https://github.com/hualet/echoflow/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-GPL--3.0-blue" alt="License"></a>
  <a href="https://github.com/hualet/echoflow/actions/workflows/build.yml"><img src="https://img.shields.io/badge/build-deepin%2025.1-brightgreen" alt="Build"></a>
</p>

---

deepin-echoflow 让 deepin 用户**直接用语音输入中文**。不需要联网、不需要云端服务、不需要手动下载模型——按下右 `Ctrl`，说话，松开，识别结果就出现在当前输入框里。

### 为什么是 deepin-echoflow

deepin-echoflow 不止是语音转文字。它直接接入 Fcitx5 输入法框架，通过原生通道把识别结果提交到当前输入上下文——不依赖剪贴板，不模拟全局按键，就像你用拼音打出一段话一样精准。所有处理都在本机完成，你的声音永远不会离开你的设备。

## 亮点

- **离线识别，隐私安全**——录音和识别全部在本机完成，不经过任何云端服务。
- **右 Ctrl 即按即说**——焦点在输入框时，按住右 `Ctrl` 开始录音，松开即结束并上屏。无需离开键盘。
- **输入框长按说话**——在输入框中按住鼠标左键约半秒开始录音，松开即转写上屏；输入框获得焦点时会在光标下方短暂提示。仅支持 X11 会话，可在设置中关闭。
- **Fcitx5 原生上屏**——文本通过 Fcitx5 的 `commitString` 提交到当前输入框，不走剪贴板，不模拟按键，永远不会打错窗口。
- **模型常驻复用**——流式识别首次录音时加载模型，后续录音和取消后重录复用权重；每次录音独立清理音频和解码状态。录音过程中后台分段转写，首次加载仍有冷启动等待。
- **录音中实时预览**——识别文字在输入框中以预编辑显示，胶囊波纹反馈实际麦克风音量。
- **应用内一键下载模型**——设置界面中点击下载，支持 hf-mirror 和 HuggingFace 官方两种下载源，下载进度条清晰可见。
- **长语音分段处理**——长句自动切分为语音段落，后台并行转写，停止录音后快速返回整段文本。
- **系统主题自适应**——基于 DTK6 的托盘图标和设置界面，自动适配 deepin 浅色/深色主题。

## 快速开始

### 环境要求

- deepin 25.1 或兼容发行版
- Fcitx5 输入法框架
- PipeWire 音频服务

### 第一步：构建

```bash
git clone https://github.com/hualet/echoflow.git deepin-echoflow
cd deepin-echoflow
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
```

### 第二步：安装

```bash
./install-user.sh
```

安装完成后会安装 `deepin-echoflow.service` 与 `deepin-echoflow.socket` 用户单元。登录时启动器导入图形会话环境、启动 UI 和录音唤醒 socket；service 仅在录音命令到达时启动，空闲 10 分钟后正常退出。
Fcitx 将焦点和光标信息直接发送到 UI socket；服务未启动时，鼠标长按仍可判断输入框位置，普通输入不会唤醒语音服务。

本地安装会同时重载 Fcitx，使用户目录中的 addon 和对应的绝对模块路径
立即生效。使用 `--no-start` 时不会修改当前 Fcitx 会话，稍后需手动执行
`fcitx5 -rd`。

通过 deb 安装或升级时，addon 使用系统模块名 `libechoflow`。由于 Debian
维护脚本不能安全重启某个用户的桌面输入法会话，安装完成后需在该用户会话中
执行一次：

```bash
fcitx5 -rd
```

### 第三步：下载模型 & 开始使用

1. 点击托盘图标 → **设置** → **模型**，选择 `Qwen3-ASR-0.6B`（轻量）或 `Qwen3-ASR-1.7B`（高精度），点击**下载**。
2. 切换到任意可输入文本的窗口，确认输入法处于激活状态。
3. 按住右 `Ctrl`，看到语音胶囊出现在屏幕中下部，开始说话。
4. 松开右 `Ctrl`，识别结果直接上屏。

## 工作流程

```
右 Ctrl 按住 ──► 开始录音 ──► Qwen ASR 实时转写 ──► 输入框预编辑预览
                                                              │
                                           松开右 Ctrl / 点击 ✓
                                                              │
                                                              ▼
                                    ┌── 后台分段最终转写 ──┐
                                    │                      │
                                    └── 合并文本 ──────────┘
                                                              │
                                                              ▼
                                              Fcitx5 提交到输入框
```

所有组件通过 Unix domain socket 在本机通信，无需网络：

| 组件 | 职责 |
| --- | --- |
| `deepin-echoflow-service` | 常驻守护进程：加载模型、管理录音、调度 ASR 转写 |
| `fcitx-addon` | Fcitx5 插件：捕获右 Ctrl 和输入焦点，接收并提交文本 |
| `deepin-echoflow-ui` | 桌面 UI：系统托盘、语音胶囊提示、设置对话框 |

## 进阶使用

### 常用运维命令

```bash
# 查看服务状态
systemctl --user status deepin-echoflow.service

# 重启已运行的服务
~/.local/bin/deepin-echoflow-session-start --restart-service-if-active

# 环境自检
./build/service/deepin-echoflow-service --self-test

# 用已有音频文件验证模型
./build/service/deepin-echoflow-service --transcribe-file your_audio.wav

# 查看默认配置
./build/service/deepin-echoflow-service --print-default-config
```

### 配置文件

`~/.config/deepin-echoflow/deepin-echoflow.conf` 中的常用项：

| 配置项 | 说明 | 默认值 |
| --- | --- | --- |
| `basic.model.model_name` | 活动模型 | `qwen3-asr-0.6b` |
| `basic.model.mirror` | 模型下载源 | `hf-mirror` |
| `basic.recording.source` | 麦克风输入源 | 自动检测 |
| `basic.recording.min_record_seconds` | 最短录音时长 | `0.5` |
| `basic.recognition.stream_transcription` | 启用流式预览 | `true` |
| `basic.recognition.stream_step_ms` | 流式转写更新间隔（毫秒，100–4000） | `500` |
| `basic.recognition.noise_suppression` | 轻量流式降噪（RNNoise） | `false` |

主要说中文时，请先在“识别语言”中明确选择 Chinese；“系统默认”会按系统语言选择，
英文桌面环境可能使用 English。不要通过润色掩盖原始识别错误。

编译降噪功能需要 `pkg-config` 和 `libsamplerate0-dev`，RNNoise 源码及内置权重随项目构建。
设置中的“降低麦克风噪声（RNNoise）”默认关闭，可在有背景噪声时开启对比。
开启后，原始录音经持久化重采样、RNNoise 小帧降噪后再送入 VAD 和 ASR；
同一录音会话复用处理状态，不等待整段录音，停止时排空降噪缓存，保留尾音和样本总数。
普通文件转写也使用同一处理器，并在进入识别前将录音采样率和声道统一转换为 16kHz 单声道，
支持设置中的 u8、s16、s32、f32 格式。波形和调试录音仍使用原始麦克风音频。
降噪可能改变发音细节，不能保证准确率提升；在安静环境中应对比后决定是否启用。
诊断日志记录有效识别语言、线程数、逐次解码耗时、当前积压、积压峰值、实际队列等待时间
和降噪总耗时，便于区分采集问题、降噪开销和识别性能不足。
`queued_ms` 表示尚未处理的音频时长，`queue_wait_ms` 表示当前取出音频包中最早样本的入队等待时间；
临时文件中的音频也保留入队时间，`queue_max_wait_ms` 汇总整次录音的最长等待。
在服务进程启动前设置 `CRISPASR_QWEN3_STREAM_PROFILE=1`，可启用 `QWEN3_STREAM_PERF` 明细，
分别记录 Mel、编码器、提示准备、LLM 预填充和文字生成耗时，以及生成 token 数、固定提示复用状态。
其中 `llm_graph_ms`、`llm_alloc_ms`、`llm_compute_ms`、`llm_transfer_ms` 是 LLM 阶段内部的分项，
不应再与外层阶段相加。诊断明细不输出音频或识别文字。
可复现的 A/B 工具和本机结果见 `docs/performance/stream-prefix-cache-2026-11-17.md`。

启用流式预览后，录音会持续送入第三方 Qwen 转写接口，按配置的音频时长触发解码，
并更新输入框预编辑文本。原生流遵循 `advanced.vad.backend`：默认 `energy` 使用能量检测，
选择 `silero` 时加载 `advanced.vad.model`，并使用配置的起始/结束阈值及最短语音、静音时长；
模型缺失或无效时会报告启动失败，不会静默退回能量检测。Silero 按不超过 100ms 的间隔分析，
配置的静音时长更短时会相应缩短分析间隔。
通常在停顿达到 `advanced.vad.min_silence_ms`（默认 500ms）后确认当前段；接近段长上限时，
若已检测到约 200ms 的停顿，则优先在此确认。连续说话时单段默认最多累计 8 秒音频。
最短语音时长加起音缓冲超过 8 秒时，段上限相应延长，避免合法语音被反复丢弃。
分段结束后关闭当前识别流并为下一段新建流，模型权重保持加载；预览更新间隔与最终分段长度独立。
采集与识别通过独立队列连接，内存中最多保存 5 秒待处理音频，更多积压转存匿名临时文件，
消费完毕或取消时自动关闭。预览间隔根据实际解码耗时调整：不短于设置值，耗时较高时
合并到约解码耗时的 1.25 倍，自动增加的间隔最多 2000ms。识别落后超过约 500ms 时，
进一步合并中间预览；等待约 1500ms 后允许再做一次预览，避免持续积压时一直停更。
这些间隔是调度预算，正在执行的推理仍需完成，不能保证固定上屏延迟。
松开时立即停止调度新的中间预览，保留全部音频，直接处理各段最终结果，
避免逐次补算过时预览。实际识别速度不足时仍可能有等待。
短于 `advanced.vad.min_speech_ms`（默认 100ms）的孤立声音段会丢弃，并撤回其临时预览；
强制切段后不足静音阈值的短间隙不会切断连续语音关系，其后的合法短尾音仍会保留；
静音达到阈值后，新的短声音段重新按最短语音时长筛选。

预览由“已确认文本 + 当前活动段”组成，后续识别只修正活动段；各段音频不重叠，
不会因为拼接去掉用户有意重复的词。录音中的正常追加和末尾少量修正立即显示；涉及更早文字的
改写等下一次解码再更新，减少说话时的整句跳动，不限制短停顿刷新或最终结果。
无声等待保留有限的起音缓冲：能量检测为 200ms，
Silero 为 1000ms，以保留模型检测起音需要的上下文。
检测到静音后暂停周期性解码；约 200ms 的短停顿会完整修正一次当前短语预览，
继续静音不会反复识别。达到配置的长停顿阈值（默认 500ms）再确认分段；
短停顿后继续说话仍保留当前段上下文，允许修正前面的词，不插入机械空格或强制标点。
短停顿修正后，如果只剩静音或稳定的低电平背景声，段落结束或松开时直接使用该结果。
如果检测到可能的轻声尾音，会保留到段落结束或松开时一并识别；恢复语音时则补回有限的间隔音频，
继续在当前段识别。
即使恰好停在预览更新边界，未做短停顿修正的当前段也会执行一次最终识别，
再将整次录音的文字统一提交。配置中显式设置的停顿时长继续生效。
能量检测每 20ms 判断，Silero 通常按 100ms 分析，检测和推理开销会叠加到短停顿更新时间。
强制分段仍可能落在词中间，跨段识别准确率需结合实际录音评估。

启用本地润色时，已确认的段落会在录音期间由独立工作线程处理，完成后逐段更新预编辑。
识别处理音频和结束排空期间优先使用计算资源：暂停新润色任务，并协作取消正在执行的任务，
识别空闲后重新处理被打断的片段；已完成的润色结果保留。
预览由“已润色确认段 + 等待润色的确认段 + 当前识别段”组成：尚未润色的部分先显示原文，
润色完成只替换对应确认段，保留后面正在识别的文字。新识别结果不会把已润色段恢复成原文。
润色预览与识别预览使用同一会话和递增版本发送；取消、切换到下一次录音、正式提交后，
晚到的润色结果都不会再改写输入框。

结束录音后先处理剩余音频，再额外等待润色最多 1500ms；已完成的段使用润色结果，未完成、
失败或超时的段使用原文。不会在结束时重复润色整篇文字，也不会在提交后用晚到的结果改写输入框。
单个后台任务有 5 秒的协作取消期限，待处理队列最多保留 8 个文字片段，超出的片段保留原文。
这项等待上限只适用于润色阶段，不包含音频积压和尾音识别；分段润色也无法处理跨段改口。

默认更新间隔为 500 毫秒，也可在设置中调整“转写更新间隔”，或在配置文件中设置。
已有配置如果显式设为 2000 毫秒，会保留 2 秒的最短更新间隔：

```ini
[basic.recognition.stream_step_ms]
value=500
```

较短间隔可更早看到预览，但会增加推理次数。当前 Qwen 实现使用当前活动段的累计音频重新识别，不沿用活动段早期转写作为生成前缀，
编码器缓存复用归一化 Mel 未变化的卷积块、相同完整输入的最终结果；
已有独立窗口注意力还可复用未变化的完整窗口。默认双向注意力在追加音频后仍重算受影响的注意力层，
不会冻结历史音频对应的 K/V。归一化变化会使相应缓存失效，缓存随识别流关闭或取消而释放。
性能日志可查看 `stem_reused`、`encoder_regions_reused` 和 `encoder_cached_bytes`，
设置 `CRISPASR_QWEN3_STREAM_ENCODER_CACHE=0` 可关闭编码器缓存。
实现范围、A/B 数据和复现方法见 `docs/performance/encoder-cache-2026-11-17.md`。
同一段音频的整段/旧前缀流式/整段修正及可选润色对照，见
`docs/performance/streaming-quality-2026-11-18.md`。
LLM 只复用音频占位符之前、跨更新不变的固定提示 KV，
这份固定提示 KV 随模型保留，可跨分段和录音复用；缓存重置、扩容或其他推理改写 KV 后自动失效。
首次预填充使用一次完整前向计算，同时生成可复用的固定提示 KV。
变化的音频及其后的提示和转写部分仍会重新计算。间隔不等于实际上屏延迟，实际速度还取决于
机器性能和当前段长度。R2T2 首段会额外等待 160ms 的前瞻音频。
第三方流接口不可用时，VAD 分段路径使用同一配置作为最长分段时长，停顿也可能提前触发。
`advanced.crisp.max_new_tokens` 为正值时，也会限制原生流每次更新和尾音处理的生成量；
设为 0 时保留后端默认预算。R2T2 会同时保留其更小的自适应预算。

识别预览以带下划线的预编辑文本显示，只有最终结果才提交到输入框；不支持行内预编辑的
应用会使用 Fcitx 输入面板预览。胶囊只显示录音波形、操作按钮及转换状态，不重复显示识别文字。
胶囊参考交互原型采用七根细柱、较紧凑的间距和录音计时，保留取消与确认操作。
波纹显示当前约 50ms 音频窗口的 FFT 频谱：去除直流分量、加 Hann 窗后合并为七个频段，
从左到右为 80–200、200–400、400–800、800–1600、1600–2600、2600–4000、4000–7600Hz；
采样率不足时，超出奈奎斯特频率的部分不显示。七根柱子一次更新，不拼接历史音量。
整体音量控制显示强度，不会把每帧频谱都拉伸为满格，也没有固定的中间高、两端低形状。
柱高上升约 40ms、回落约 150ms；静音时保持短条。录音计时不因状态刷新重置。
静音显示结合固定能量门限、自适应背景噪声估计、频谱平坦度和 RNNoise 语音概率，
不完全依赖一个语音概率阈值，以保留轻声反馈。该处理仅控制波形，不拦截识别音频，
也不依赖是否开启识别降噪，实际效果仍需结合麦克风环境验证。
停止、取消或超过 250ms 未收到音量更新时归零；静音消息会清空整帧柱高。
流式转写和普通文件录音均支持，音量更新不等待 ASR 识别结果。
松开按键后，输入框中的预编辑继续接收尾音和润色更新，直到插件处理最终提交时才结束，
不再先单独清空。取消时仍会清理预览。
若结束识别发生异常，服务会保留最后一份识别预览并尝试提交；已撤回的噪声预览不会恢复，
主动取消和焦点校验仍优先，不会因为异常回退而提交到其他输入框。
UI 通知使用非阻塞发送，避免通知发送阻塞识别、结束或取消操作。
音量使用独立的 `deepin-echoflow-ui-level.sock`，与 `deepin-echoflow-ui.sock` 的会话状态队列分离；
音量队列拥塞时直接丢弃该次采样，不占用开始、转换和结束状态的队列。
UI 优先处理状态消息，积压音量只显示最新值，新一轮录音会丢弃上一轮的积压音量。
更新插件源码后需重新安装并重启 Fcitx
（`./install-user.sh` 会完成这两步）。

界面品牌名称保留 **EchoFlow**，包括窗口标题、托盘提示、使用向导和桌面显示名称。
项目包名、程序、插件、桌面文件名及 systemd 服务统一使用 `deepin-echoflow`。
新配置位于 `~/.config/deepin-echoflow/deepin-echoflow.conf`，默认录音目录为
`~/.local/share/deepin-echoflow/recordings`。升级时若只有旧的
`~/.config/echoflow/echoflow.conf`，会继续读取该文件及同目录下的模型，无需重新下载；
新旧配置同时存在时优先使用新配置。`install-user.sh` 会停止旧服务并清理旧启动入口。

修改配置后，重启已运行的服务即可生效；未运行时，下次唤醒会加载新配置：

```bash
~/.local/bin/deepin-echoflow-session-start --restart-service-if-active
```

### 原音频与降噪识别对比

准备包含安静和噪声录音的本地标注清单（16kHz、单声道、PCM16 WAV）：

```json
[
  {"id": "clean-zh", "audio": "clean.wav", "reference": "今天下午三点开会", "condition": "clean"},
  {"id": "noisy-zh", "audio": "noisy.wav", "reference": "今天下午三点开会", "condition": "noisy"}
]
```

运行对比（音频路径相对于清单文件；模型路径使用实际已下载文件）：

```bash
python3 scripts/compare-denoise.py \
  --binary build/service/deepin-echoflow-service \
  --model ~/.config/deepin-echoflow/qwen3-asr-0.6b/qwen3-asr-0.6b-q4_k.gguf \
  --manifest /path/to/manifest.json --output /tmp/denoise-comparison.json
```

工具固定使用同一音频、同一模型、同一语言，关闭润色，每项默认运行三次；
按环境分别报告原始转写、降噪转写、中文字符错误率（忽略空白及标点）和耗时。
报告中的耗时包含每次启动时的模型加载，不能作为实时预览延迟；实时性能请看服务日志。
模型仍位于旧配置目录时，`--model` 指向原来的文件即可，无需移动或重新下载。

### 卸载

```bash
./uninstall-user.sh
```

卸载会保留 `~/.config/deepin-echoflow` 和 `~/.local/share/deepin-echoflow` 下的用户数据和模型。

## 开发者

```bash
# 运行测试
ctest --test-dir build --output-on-failure

# 从构建目录运行
./run.sh

# 直接启动 UI
./build/ui-host/deepin-echoflow-ui
```

默认 CTest 和 Debian 打包测试不运行依赖 Git 的发布工具测试。
发布工具修改后，可在安装了 Git 的开发环境中单独验证：

```bash
python3 -m unittest -v tests.test_release_performance_driver tests.test_release_evidence
```

源码结构：

```
service/         守护进程、录音管线、ASR 引擎、状态机
fcitx-addon/     Fcitx5 插件（右 Ctrl 捕获、文本提交）
ui-host/         Qt6/DTK6 托盘和设置界面
qml/             语音胶囊 QML 动画
qwen-asr-runtime/  CMake 封装 qwen-asr C 运行时
tests/           QTest 逻辑测试、Shell spec 检查、性能基准
```

测试约束：逻辑测试不依赖模型权重、PipeWire 录音或运行中的 Fcitx。

### 发布性能基线

每次提升版本号前运行性能门禁。若上次记录的 CPU、模型、样本、配置和线程
设置均与当前机器一致，脚本复用历史基线；否则在当前机器以相同参数构建基线
版本和候选版本后再比较：

```bash
python3 scripts/check-release-performance.py \
  --baseline-ref v0.2.1 \
  --candidate-ref HEAD \
  --version 0.2.2 \
  --config "$HOME/.config/deepin-echoflow/deepin-echoflow.conf" \
  --model /path/to/model-or-model-directory \
  --manifest /path/to/release-manifest.json \
  --iterations 3 \
  --output docs/performance/releases/0.2.2.json
```

有历史 evidence 时增加
`--previous-evidence docs/performance/releases/<上一版本>.json`。聚合中位延迟
最多回退 10%，单样本最多回退 20%，CER 不得增加，必需样本不得产生空转写。

发布检查还会按 20 ms 一帧用实际录音速度回放同一批 16 kHz 音频，走服务的原生流式
识别、VAD 和降噪路径。另记首条非空预编辑、停止到最终识别结果的耗时及原始流式 CER；
不含冷启动、LLM 润色或 Fcitx 提交。样本应包含至少一段能在松键前出现预编辑的录音，
短样本可无预编辑。上个版本没有流式基准时，继续比较整文件指标，并建立首份流式基线；
两边都有流式数据时，按相同阈值比较，原本稳定出现的预编辑不得消失。

## 鸣谢

- [Fcitx5](https://github.com/fcitx/fcitx5)——输入法框架
- [qwen-asr](https://github.com/antirez/qwen-asr)——纯 C 的 Qwen ASR 运行时
- [Qwen](https://github.com/QwenLM/Qwen)——Qwen ASR 模型
- [xzl01/qwen-voice-input](https://github.com/xzl01/qwen-voice-input)——灵感来源

## License

GPL-3.0-or-later
