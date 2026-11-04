"""文本工具：消息内容抽取、中文友好的分词与相似度。

中文没有空格分词，这里用“字符二元组 + 英文/数字单词”作为检索与比对的基本单位，
不依赖任何分词库，足以支撑轻量检索与事实去重。
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from typing import Any

_CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]+")
_WORD_RE = re.compile(r"[a-z0-9]+")
_TOKEN_SPLIT_RE = re.compile(r"([㐀-䶿一-鿿豈-﫿]+|[a-z0-9]+)")

# 高频但几乎不携带信息的二元组，检索时降权
STOP_BIGRAMS = frozenset(
    """怎么 什么 我们 你们 他们 她们 一个 这个 那个 没有 就是 可以 现在 然后 因为 所以 但是
    还是 已经 觉得 知道 时候 一下 一点 有点 是不 不是 的话 这样 那样 这么 那么 为什么 吗你
    你的 我的 他的 她的 好的 好吧 嗯嗯 哈哈 呵呵 了吗 了吧 了啊 一起 自己 东西 事情 真的""".split()
)


def content_to_text(content: Any) -> str:
    """把 OpenAI 消息 content（字符串或多模态分段列表）转成纯文本。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                ptype = part.get("type")
                if ptype == "text":
                    parts.append(str(part.get("text", "")))
                elif ptype in ("image_url", "input_image", "image"):
                    parts.append("[图片]")
                elif ptype in ("input_audio", "audio"):
                    parts.append("[语音]")
        return "".join(parts)
    return str(content)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    return text


def tokens(text: str) -> list[str]:
    """字符二元组 + 英文/数字单词。单个汉字的片段保留为单字。"""
    out: list[str] = []
    for run in _TOKEN_SPLIT_RE.findall(normalize(text)):
        if _CJK_RE.fullmatch(run):
            if len(run) == 1:
                out.append(run)
            else:
                out.extend(run[i : i + 2] for i in range(len(run) - 1))
        else:
            out.append(run)
    return out


def token_set(text: str) -> set[str]:
    return set(tokens(text))


def dice(a: str, b: str) -> float:
    """两段文本的二元组 Dice 系数，0~1。"""
    na, nb = _compact(a), _compact(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    sa, sb = token_set(na), token_set(nb)
    if not sa or not sb:
        return 0.0
    return 2 * len(sa & sb) / (len(sa) + len(sb))


def _compact(text: str) -> str:
    return "".join(_TOKEN_SPLIT_RE.findall(normalize(text)))


def same_value(a: str, b: str, threshold: float) -> bool:
    ca, cb = _compact(a), _compact(b)
    if not ca or not cb:
        return False
    if ca == cb or (len(ca) >= 2 and len(cb) >= 2 and (ca in cb or cb in ca)):
        return True
    return dice(ca, cb) >= threshold


def truncate(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(1, limit - 1)] + "…"


class LexicalIndex:
    """在一小批文档上做 IDF 加权的“查询覆盖率”打分，结果落在 0~1，可设绝对阈值。"""

    def __init__(self, docs: Iterable[tuple[str, str]]):
        self.doc_tokens: dict[str, set[str]] = {}
        df: Counter[str] = Counter()
        for doc_id, text in docs:
            toks = token_set(text)
            self.doc_tokens[doc_id] = toks
            df.update(toks)
        self.n = max(1, len(self.doc_tokens))
        self.df = df

    def idf(self, tok: str) -> float:
        # 语料中没出现过的词按 df=1 计权：否则在小语料上，查询里的“生词”权重最大，
        # 命中词的覆盖率会被不合理地压低
        d = max(1, self.df.get(tok, 0))
        weight = math.log(1 + (self.n - d + 0.5) / (d + 0.5))
        if tok in STOP_BIGRAMS:
            weight *= 0.2
        return weight

    def score(self, query: str) -> dict[str, float]:
        q = token_set(query)
        if not q:
            return {}
        weights = {t: self.idf(t) for t in q}
        total = sum(weights.values()) or 1.0
        scores: dict[str, float] = {}
        for doc_id, toks in self.doc_tokens.items():
            hit = sum(w for t, w in weights.items() if t in toks)
            if hit > 0:
                scores[doc_id] = hit / total
        return scores


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
