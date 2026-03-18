from __future__ import annotations
import base64
import inspect
import json
import re
import time
from pathlib import Path
from openai import AsyncOpenAI
from astrbot.api import logger
from .image_format import guess_image_mime_and_ext
from .openai_compat_backend import (
    build_proxy_http_client,
    normalize_openai_compat_base_url,
)

_MARKDOWN_IMAGE_RE = re.compile(r"!\[.*?\]\((.*?)\)")
_DATA_IMAGE_RE = re.compile(r"(data:image/[^\s)]+)")
_HTML_IMG_RE = re.compile(r'<img[^>]*src=["\']([^"\'>]+)["\']', re.IGNORECASE)
_IMAGE_URL_RE = re.compile(
    r"(https?://[^\s<>\"')\]]+?\.(?:png|jpg|jpeg|webp|gif)(?:\?[^\s<>\"')\]]*)?)",
    re.IGNORECASE,
)
_JSON_URL_FIELD_RE = re.compile(
    r'"(?:image_url|imageUrl|url|image|src|uri|link|href|fifeUrl|fife_url|final_image_url|origin_image_url)"\s*:\s*"([^"]+)"'
)
_HTML_VIDEO_RE = re.compile(r'<video[^>]*src=["\']([^"\'>]+)["\']', re.IGNORECASE)
_VIDEO_URL_RE = re.compile(
    r"(https?://[^\s<>\"')\]]+?\.(?:mp4|webm|mov)(?:\?[^\s<>\"')\]]*)?)",
    re.IGNORECASE,
)
_BASE64_PREFIX_RE = re.compile(r"^(?:b64|base64)\s*:\s*", re.IGNORECASE)


def _strip_markdown_target(target: str) -> str | None:
    s = (target or "").strip()
    if not s:
        return None
    if s.startswith("<") and ">" in s:
        right = s.find(">")
        if right > 1:
            s = s[1:right].strip()
    m = re.match(r'^(?P<url>\S+)(?:\s+(?:"[^"]*"|\'[^\']*\'))?\s*$', s)
    if m:
        s = m.group("url")
    s = s.strip().strip('"').strip("'")
    return s or None


def _decode_base64_bytes(text: str) -> bytes:
    s = re.sub(r"\s+", "", str(text or "").strip())
    if not s:
        return b""
    candidates = [s, s.replace("-", "+").replace("_", "/")]
    for cand in candidates:
        pad = "=" * ((4 - len(cand) % 4) % 4)
        try:
            raw = base64.b64decode(cand + pad, validate=False)
            if raw:
                return raw
        except Exception:
            continue
    try:
        raw = base64.urlsafe_b64decode(s + ("=" * ((4 - len(s) % 4) % 4)))
        if raw:
            return raw
    except Exception:
        pass
    return b""


def _looks_like_video_url(url: str) -> bool:
    u = (url or "").strip().lower()
    if not u.startswith(("http://", "https://")):
        return False
    if any(ext in u for ext in (".mp4", ".webm", ".mov")):
        return True
    if "generated_video" in u:
        return True
    return False


def _is_valid_data_image_ref(ref: str) -> bool:
    s = str(ref or "").strip()
    if not s.startswith("data:image/"):
        return False
    if "," not in s:
        return False
    _header, b64 = s.split(",", 1)
    b64 = re.sub(r"\s+", "", (b64 or "").strip())
    if not b64 or b64 == "...":
        return False
    if len(b64) < 16:
        return False
    try:
        if not re.fullmatch(r"[A-Za-z0-9+/=_-]+", b64[:2048]):
            return False
    except Exception:
        pass
    if len(b64) < 128:
        raw = _decode_base64_bytes(b64)
        if not raw:
            return False
        if not _guess_mime_from_magic(raw):
            return False
    return True


def _guess_mime_from_magic(image_bytes: bytes) -> str | None:
    if len(image_bytes) >= 3 and image_bytes[0:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if len(image_bytes) >= 8 and image_bytes[0:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(image_bytes) >= 6 and (image_bytes[0:6] in (b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(image_bytes) >= 12 and image_bytes[0:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return None


def _base64_to_data_image_ref(text: str) -> str | None:
    s = (text or "").strip().strip('"').strip("'")
    s = _BASE64_PREFIX_RE.sub("", s).strip()
    s = re.sub(r"\s+", "", s)
    if len(s) < 128:
        return None
    raw = _decode_base64_bytes(s)
    if not raw:
        return None
    mime = _guess_mime_from_magic(raw)
    if not mime:
        return None
    std_b64 = base64.b64encode(raw).decode()
    return f"data:{mime};base64,{std_b64}"


def _extract_first_image_ref(text: str) -> str | None:
    s = (text or "").strip()
    if not s:
        return None
    if s.startswith("data:image/"):
        compact = re.sub(r"\s+", "", s)
        if _is_valid_data_image_ref(compact):
            return compact
    m = _MARKDOWN_IMAGE_RE.search(s)
    if m:
        cand = _strip_markdown_target(m.group(1))
        if cand:
            if cand.startswith("data:image/"):
                cand = re.sub(r"\s+", "", cand)
                if _is_valid_data_image_ref(cand):
                    return cand
            elif not _looks_like_video_url(cand):
                return cand
    for m in _DATA_IMAGE_RE.finditer(s):
        cand = re.sub(r"\s+", "", m.group(1).strip())
        if _is_valid_data_image_ref(cand):
            return cand
    m = _HTML_IMG_RE.search(s)
    if m:
        url = m.group(1).strip()
        if url and not _looks_like_video_url(url):
            return url
    m = _IMAGE_URL_RE.search(s)
    if m:
        url = m.group(1).strip()
        if url and not _looks_like_video_url(url):
            return url
    if s.startswith("http://") or s.startswith("https://"):
        if _looks_like_video_url(s):
            return None
        return s
    for m in _JSON_URL_FIELD_RE.finditer(s):
        cand = m.group(1).strip().replace("\\/", "/")
        cand = _strip_markdown_target(cand) or cand
        if cand.startswith("data:image/"):
            cand = re.sub(r"\s+", "", cand)
            if _is_valid_data_image_ref(cand):
                return cand
        if cand.startswith(("http://", "https://")) and not _looks_like_video_url(cand):
            return cand
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            parsed = json.loads(s)
        except Exception:
            parsed = None
        if parsed is not None:
            for v in _iter_strings(parsed):
                ref = _extract_first_image_ref(v)
                if ref:
                    return ref
    ref = _base64_to_data_image_ref(s)
    if ref:
        return ref
    return None


def _extract_first_video_url(text: str) -> str | None:
    s = (text or "").strip()
    if not s:
        return None
    m = _HTML_VIDEO_RE.search(s)
    if m:
        url = m.group(1).strip()
        return url if _looks_like_video_url(url) else None
    m = _VIDEO_URL_RE.search(s)
    if m:
        url = m.group(1).strip()
        return url if _looks_like_video_url(url) else None
    if _looks_like_video_url(s):
        return s
    return None


def _is_client_closed_error(exc: Exception) -> bool:
    msg = f"{exc!r} {exc}".lower()
    if "client has been closed" in msg:
        return True
    cur: Exception | None = exc
    for _ in range(3):
        nxt = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
        if not isinstance(nxt, Exception):
            break
        cur = nxt
        if "client has been closed" in f"{cur!r} {cur}".lower():
            return True
    return False


async def _resolve_awaitable(value: object) -> object:
    while inspect.isawaitable(value):
        value = await value
    return value


def _iter_strings(obj: object) -> list[str]:
    out: list[str] = []
    seen: set[int] = set()
    def walk(x: object) -> None:
        if x is None:
            return
        oid = id(x)
        if oid in seen:
            return
        seen.add(oid)
        if isinstance(x, str):
            out.append(x)
            return
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
            return
        if isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
            return
        model_dump = getattr(x, "model_dump", None)
        if callable(model_dump):
            try:
                walk(model_dump())
                return
            except Exception:
                pass
        as_dict = getattr(x, "dict", None)
        if callable(as_dict):
            try:
                walk(as_dict())
                return
            except Exception:
                pass
        obj_dict = getattr(x, "__dict__", None)
        if isinstance(obj_dict, dict):
            walk(obj_dict)
            return
    walk(obj)
    return out


def _extract_image_ref_from_content(content: object) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        return _extract_first_image_ref(content)
    if isinstance(content, list):
        for part in content:
            ref = _extract_image_ref_from_content(part)
            if ref:
                return ref
        return None
    if isinstance(content, dict):
        # ...（保持原样，省略以节省篇幅，但实际复制时请保留你原来的这段）
        pass  # 这里保持你原来的 _extract_image_ref_from_content 内容
    for s in _iter_strings(content):
        ref = _extract_first_image_ref(s)
        if ref:
            return ref
    return None


def _extract_video_ref_from_content(content: object) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        return _extract_first_video_url(content)
    for s in _iter_strings(content):
        url = _extract_first_video_url(s)
        if url:
            return url
    return None


def _extract_media_refs_from_sse_text(text: str) -> tuple[list[str], list[str]]:
    # 保持原样
    image_refs: list[str] = []
    video_refs: list[str] = []
    full_text = ""
    # ...（保持你原来的这个函数）
    return image_refs, video_refs


class OpenAIChatImageBackend:
    """Image generation/edit via chat.completions (gateway-style)."""

    def __init__(self, *, imgr, base_url: str, api_keys: list[str], timeout: int = 120,
                 max_retries: int = 2, default_model: str = "", supports_edit: bool = True,
                 extra_body: dict | None = None, proxy_url: str | None = None):
        self.imgr = imgr
        self.base_url = normalize_openai_compat_base_url(base_url)
        self.api_keys = [str(k).strip() for k in (api_keys or []) if str(k).strip()]
        self.timeout = int(timeout or 120)
        self.max_retries = int(max_retries or 2)
        self.default_model = str(default_model or "").strip()
        self.supports_edit = bool(supports_edit)
        self.extra_body = extra_body or {}
        self.proxy_url = str(proxy_url or "").strip() or None
        self._key_index = 0
        self._clients: dict[str, AsyncOpenAI] = {}
        self._http_client = None

    # ...（_supports_http_client_param, _get_http_client, close, _next_key, _get_client, _recreate_client, _normalize_ref_candidate, _extract_image_refs_from_response 等所有方法保持不变）

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        size: str | None = None,
        resolution: str | None = None,
        extra_body: dict | None = None,
    ) -> Path:
        key = self._next_key()
        client = self._get_client(key)
        final_model = str(model or self.default_model or "").strip()
        if not final_model:
            raise RuntimeError("未配置 model")

        user_text = (
            f"{prompt}\n\n"
            "Immediately return ONLY this exact line, nothing else:\n"
            "![Generated Image](https://your-direct-image-url-here)\n"
            "Use direct https URL. Do NOT split."
        )

        eb = {**self.extra_body, **(extra_body or {})}
        t0 = time.time()
        full_content = ""

        try:
            stream = await client.chat.completions.create(
                model=final_model,
                messages=[{"role": "user", "content": user_text}],
                extra_body=eb or None,
                stream=True,
            )
            logger.info("[generate] 开始收集 streaming chunks...")

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    chunk_text = chunk.choices[0].delta.content
                    full_content += chunk_text
                    logger.debug("[chunk] %s", chunk_text.strip()[:80])

            logger.info("[generate] 完整 content 长度: %d", len(full_content))
            logger.info("[generate] 完整 content 前1000: %s", full_content[:1000] or "空")

        except Exception as e:
            if _is_client_closed_error(e):
                client = await self._recreate_client(key)
                stream = await client.chat.completions.create(..., stream=True)
                full_content = ""
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        full_content += chunk.choices[0].delta.content
            else:
                raise

        # 提取图片
        ref = _extract_first_image_ref(full_content)
        if not ref:
            # 强力 fallback：专门抓 hcg.pippi.top 的 Google Storage 链接
            url_match = re.search(r'https?://storage\.googleapis\.com/[^\s"\'()<>]+', full_content)
            if url_match:
                ref = url_match.group(0)
                logger.info("[generate] 成功抓到 Google Storage 直链！ref = %s", ref)

        if not ref:
            raise RuntimeError(f"仍未提取到图片！完整 content 前500:\n{full_content[:500]}")

        return await self._save_from_ref(ref, debug_snippet=full_content[:300])

    async def edit(
        self,
        prompt: str,
        images: list[bytes],
        *,
        model: str | None = None,
        size: str | None = None,
        resolution: str | None = None,
        extra_body: dict | None = None,
    ) -> Path:
        if not self.supports_edit:
            raise RuntimeError("该后端不支持改图")
        if not images:
            raise ValueError("至少需要一张图片")

        key = self._next_key()
        client = self._get_client(key)
        final_model = str(model or self.default_model or "").strip()
        if not final_model:
            raise RuntimeError("未配置 model")

        text = (
            f"{prompt}\n\n"
            "Edit the attached image(s). Return ONLY this exact line:\n"
            "![Edited Image](https://your-direct-image-url-here)\n"
            "Use direct https URL. Do NOT split."
        )

        parts = [{"type": "text", "text": text}]
        for img_bytes in images:
            mime, _ = guess_image_mime_and_ext(img_bytes)
            parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(img_bytes).decode()}"}})

        eb = {**self.extra_body, **(extra_body or {})}
        t0 = time.time()
        full_content = ""

        try:
            stream = await client.chat.completions.create(
                model=final_model,
                messages=[{"role": "user", "content": parts}],
                extra_body=eb or None,
                stream=True,
            )
            logger.info("[edit] 开始收集 streaming chunks...")

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    full_content += chunk.choices[0].delta.content

            logger.info("[edit] 完整 content 长度: %d", len(full_content))
            logger.info("[edit] 完整 content 前1000: %s", full_content[:1000] or "空")

        except Exception as e:
            if _is_client_closed_error(e):
                client = await self._recreate_client(key)
                stream = await client.chat.completions.create(..., stream=True)
                full_content = ""
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        full_content += chunk.choices[0].delta.content
            else:
                raise

        ref = _extract_first_image_ref(full_content)
        if not ref:
            url_match = re.search(r'https?://storage\.googleapis\.com/[^\s"\'()<>]+', full_content)
            if url_match:
                ref = url_match.group(0)
                logger.info("[edit] 成功抓到 Google Storage 直链！")

        if not ref:
            raise RuntimeError(f"仍未提取到图片！完整 content 前500:\n{full_content[:500]}")

        return await self._save_from_ref(ref, debug_snippet=full_content[:300])

    # 其余方法（_save_single_ref、_save_from_ref、_extract_image_ref_from_response 等）保持你原来的代码不变
