#!/usr/bin/env python
"""Generate tests/data/pinyin_parity.tsv — reference g2p outputs for the
F5-TTS Chinese g2p parity test (#294).

Each line: <input>\t<tok1>\x1f<tok2>\x1f...  where the token list is exactly
F5-TTS's convert_char_to_pinyin(input)[0]. The C++ test-pinyin-g2p compares
core_pinyin::convert_char_to_pinyin against these.

Run: python tests/gen_pinyin_parity.py   (requires pypinyin + f5_tts)
"""
from pathlib import Path
import sys

SENTENCES = [
    # polyphones (context changes the reading)
    "银行", "行走", "自行车", "长城", "成长", "长大", "校长",
    "的确", "目的", "我的书", "了解", "为了", "他走了",
    "得到", "觉得", "我得走了", "音乐", "快乐", "重要", "重复",
    "还是", "还钱", "教育", "教书", "差不多", "参差",
    "中国", "中间", "看中", "都是", "首都", "地方", "土地",
    # 不 sandhi
    "不是", "不对", "不好", "不能", "不要", "我不知道", "是不是", "不大不小",
    # 一 sandhi
    "一个", "一天", "一年", "一起", "一定", "第一", "一一", "看一看", "统一", "一二三",
    # third-tone sandhi
    "你好", "老鼠", "保管好", "我很好", "买雨伞", "勇敢", "领导", "管理", "水果", "理想很美好",
    # sentences
    "我们是中国人。", "今天天气很好。", "他昨天买了一本书。",
    "请问洗手间在哪里？", "这个多少钱？", "谢谢你的帮助！",
    "我爱北京天安门。", "学习使我快乐。", "时间就是金钱。",
    "中华人民共和国", "人工智能正在改变世界。", "他一个人走了很长的路。",
    "不好意思，请再说一遍。", "我一定会努力工作。",
    # mixed CN + EN / digits / punctuation
    "Hello 世界", "AI 人工智能 is cool.", "我在 Beijing 工作。",
    "价格是 100 元。", "他说：“你好，世界。”", "第3个问题很难。",
    "这是 GPT 模型。", "我有 2 个苹果和 3 个梨。",
    # punctuation-heavy / edge
    "你好，世界！", "真的吗？！", "一，二，三。",
    "长长的头发", "好好学习，天天向上。",
    # real-world prose (news / conversation), less adversarial
    "今天上午，国家主席在北京会见了来访的外国代表团。",
    "根据最新的天气预报，明天将有大雨，请大家注意安全。",
    "这家公司去年的销售额比前年增长了百分之二十。",
    "我昨天去了超市，买了一些水果和蔬菜。",
    "他每天早上六点起床，然后去公园跑步。",
    "科学家们发现了一种新的治疗方法，可能会挽救很多生命。",
    "请你把这份文件发给所有的同事，谢谢。",
    "虽然工作很忙，但是他从来没有抱怨过。",
    "孩子们在草地上快乐地玩耍，笑声传遍了整个花园。",
    "这部电影讲述了一个感人的爱情故事。",
    "会议将于下周三下午三点在二楼会议室举行。",
    "学习一门新的语言需要时间和耐心。",
    "我们应该保护环境，减少污染，节约资源。",
    "他的房间里放满了各种各样的书。",
    "经过多年的努力，他终于实现了自己的梦想。",
]

SEP = "\x1f"


def main():
    from f5_tts.model.utils import convert_char_to_pinyin
    out = Path("tests/data/pinyin_parity.tsv")
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for s in SENTENCES:
        toks = convert_char_to_pinyin([s])[0]
        assert "\t" not in s and SEP not in s
        lines.append(s + "\t" + SEP.join(toks))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(lines)} sentences", file=sys.stderr)


if __name__ == "__main__":
    main()
