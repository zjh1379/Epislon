"""枫云 AI 助手 ⇄ 大脑代理 的辅助函数（放进枫云目录即可 import）。

枫云社区版的 chat_llm 是非流式调用：要等整段回复生成完，才开始切句合成语音。
本模块提供“边生成边切句”的工具，供第 6 步（体验调优）改造成流式时使用：

    from her_bridge import her_metadata, stream_sentences
    for sentence in stream_sentences(client, model, messages, her_metadata(username, voiceprint_switch)):
        tts_queue.put(sentence)        # 由单独的播放线程按顺序合成、播放

只依赖 openai SDK（枫云已安装）。切句逻辑是纯函数，见 split_stream。
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from typing import Any

# 句末标点：遇到就立即送 TTS
HARD_BREAK = re.compile(r"[。！？!?；;…\n]+[”」』）)]*")
# 次级停顿：当缓冲已经够长时，在逗号处提前切，缩短首句等待
SOFT_BREAK = re.compile(r"[，,、：:]")


def her_metadata(username: str, voiceprint_switch: str, *, external_user_id: str = "owner") -> dict[str, Any]:
    """枫云语音入口的身份元数据。声纹识别开启时，非主人的语音在识别阶段就被丢弃，因此视为已确认。"""
    return {
        "channel": "voice",
        "scene": "voice",
        "external_user_id": external_user_id,
        "user_name": username,
        "speaker_verified": voiceprint_switch == "开启",
    }


def split_stream(chunks: Iterable[str], *, first_min: int = 6, soft_min: int = 18, max_len: int = 60) -> Iterator[str]:
    """把流式文本切成适合 TTS 的句子。

    - 遇到句末标点立即输出；
    - 第一句只要达到 first_min 个字、遇到逗号就输出（越早开口越好）；
    - 之后的句子在超过 soft_min 字时可在逗号处切；超过 max_len 强制切开。
    """
    buf = ""
    emitted = 0
    for chunk in chunks:
        if not chunk:
            continue
        buf += chunk
        while buf:
            m = HARD_BREAK.search(buf)
            if m:
                sentence, buf = buf[: m.end()], buf[m.end() :]
                if sentence.strip():
                    emitted += 1
                    yield sentence.strip()
                continue
            limit = first_min if emitted == 0 else soft_min
            soft = None
            for sm in SOFT_BREAK.finditer(buf):
                if sm.end() >= limit:
                    soft = sm
                    break
            if soft is not None:
                sentence, buf = buf[: soft.end()], buf[soft.end() :]
                emitted += 1
                yield sentence.strip()
                continue
            if len(buf) >= max_len:
                sentence, buf = buf[:max_len], buf[max_len:]
                emitted += 1
                yield sentence.strip()
                continue
            break
    if buf.strip():
        yield buf.strip()


def stream_sentences(client: Any, model: str, messages: list[dict[str, Any]], metadata: dict[str, Any],
                     **kwargs: Any) -> Iterator[str]:
    """调用大脑代理（流式），逐句产出回复。client 为 openai.OpenAI 实例。"""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        extra_body={"metadata": metadata, "enable_thinking": False},
        **kwargs,
    )

    def _texts() -> Iterator[str]:
        for chunk in resp:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    yield from split_stream(_texts())
