"""
LLM客户端封装
统一使用OpenAI格式调用
"""

import json
import re
from typing import Optional, Dict, Any, List, Tuple
from openai import OpenAI

from ..config import Config


def _extract_response(response) -> Tuple[str, Optional[str]]:
    """从 OpenAI 响应对象中安全提取文本内容和 finish_reason。

    不同提供商/SDK 版本字段略有差异，这里做兼容处理。
    """
    content = ""
    finish_reason = None
    try:
        choice = response.choices[0]
        # content 可能是 str，也可能是 None（部分工具调用场景），也可能是 list
        msg = getattr(choice, 'message', None)
        if msg is not None:
            raw = getattr(msg, 'content', None)
            if isinstance(raw, str):
                content = raw
            elif raw is None:
                content = ""
            else:
                # 某些兼容层返回 list[{"type":"text","text":...}]
                try:
                    content = "".join(
                        p.get("text", "") for p in raw
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
                except Exception:
                    content = str(raw)
        finish_reason = getattr(choice, 'finish_reason', None)
    except Exception:
        # 兜底：尝试字典访问
        try:
            choice = response['choices'][0]
            msg = choice.get('message', {})
            content = msg.get('content') or ""
            finish_reason = choice.get('finish_reason')
        except Exception:
            pass
    return content, finish_reason


def _strip_thinking(content: str) -> str:
    """剥离推理类模型的 <think>...</think> 思考链。

    处理三种情况：
    1. 正常闭合的 <think>...</think>
    2. 被截断未闭合的 <think>...（输出被 max_tokens 砍掉末尾）
    3. 文本前导的纯空白
    """
    # 先处理闭合的情况
    content = re.sub(r'<think>[\s\S]*?</think>', '', content)
    # 再处理未闭合的情况（被截断）：从 <think> 起一直删到结尾
    content = re.sub(r'<think>[\s\S]*$', '', content)
    return content.strip()


class LLMClient:
    """LLM客户端"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME

        if not self.api_key:
            raise ValueError("LLM_API_KEY 未配置")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        发送聊天请求

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数。None 时使用 Config.LLM_MAX_TOKENS
            response_format: 响应格式（如JSON模式）

        Returns:
            模型响应文本（已剥离 <think> 推理链）
        """
        # 默认使用 Config 中较大的 token 预算，避免推理类模型被截断
        if max_tokens is None:
            max_tokens = Config.LLM_MAX_TOKENS

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format:
            kwargs["response_format"] = response_format

        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            # 部分兼容层（如 Google Gemini 的 OpenAI 适配端点）不接受 response_format
            # 或 temperature 等 OpenAI 专有参数，会以 400/500 失败。
            # 这里做一次降级重试：去掉 response_format 后再试。
            msg = str(e).lower()
            if response_format and ('response_format' in msg or 'json' in msg
                                    or '400' in msg or '500' in msg
                                    or 'internal' in msg or 'invalid' in msg):
                kwargs.pop('response_format', None)
                response = self.client.chat.completions.create(**kwargs)
            else:
                raise

        content, finish_reason = _extract_response(response)

        # 推理类模型（如 deepseek-v4-flash-free、MiniMax M2 等）会先输出
        # <think>...</think> 推理链。如果输出被 max_tokens 截断，结束标签可能缺失。
        # 这里同时处理「正常闭合」与「被截断未闭合」两种情况。
        content = _strip_thinking(content)

        # 被截断的输出几乎无法解析为合法 JSON，提前给出清晰错误，避免下游"JSON 格式无效"误导
        if finish_reason == 'length':
            raise ValueError(
                f"LLM 输出被 max_tokens={max_tokens} 截断（finish_reason=length）。"
                f"推理类模型会消耗大量 token 在 <think> 链上，请调大环境变量 LLM_MAX_TOKENS。"
                f"当前返回内容长度: {len(content)} 字符。"
            )

        return content
    
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        发送聊天请求并返回JSON

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数。None 时使用 Config.LLM_MAX_TOKENS

        Returns:
            解析后的JSON对象
        """
        if max_tokens is None:
            max_tokens = Config.LLM_MAX_TOKENS

        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        # 清理markdown代码块标记
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            raise ValueError(f"LLM返回的JSON格式无效: {cleaned_response[:500]}")

