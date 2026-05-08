"""Async client tới vLLM server (OpenAI-compatible).

Mỗi vLLM upstream có pool kết nối riêng để tránh head-of-line blocking giữa 2 model
(theo phản biện #15 Backend Architect).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import httpx

from utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class VllmEndpoint:
    name: str  # "translator" | "ocr"
    base_url: str
    served_model_name: str
    connect_timeout_s: float = 3.0
    request_timeout_s: float = 60.0
    pool_max_keepalive: int = 20
    pool_max_connections: int = 50

    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)
    _model_fingerprint: str | None = field(default=None, init=False, repr=False)

    async def start(self) -> None:
        if self._client is not None:
            return
        timeout = httpx.Timeout(
            connect=self.connect_timeout_s,
            read=self.request_timeout_s,
            write=5.0,
            pool=2.0,
        )
        limits = httpx.Limits(
            max_keepalive_connections=self.pool_max_keepalive,
            max_connections=self.pool_max_connections,
        )
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            limits=limits,
            http2=False,
        )
        log.info("vllm_client_started", endpoint=self.name, base_url=self.base_url)

    async def stop(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(f"vLLM client {self.name} chưa được start")
        return self._client

    async def list_models(self) -> dict[str, Any]:
        resp = await self.client.get("/v1/models")
        resp.raise_for_status()
        return resp.json()

    async def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = await self.client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def deep_health_check(self) -> dict[str, Any]:
        """Verify model thật sự ready bằng inference call tối thiểu.

        Theo #27 SRE: container up != model ready. Phải gọi /v1/chat/completions
        thật để xác nhận weight đã load và inference path hoạt động.
        """
        payload = {
            "model": self.served_model_name,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "temperature": 0,
        }
        try:
            data = await asyncio.wait_for(self.chat_completion(payload), timeout=5.0)
            return {"ok": True, "model": data.get("model"), "id": data.get("id")}
        except (httpx.HTTPError, asyncio.TimeoutError) as exc:
            return {"ok": False, "error": str(exc)[:200]}

    async def get_model_fingerprint(self) -> str:
        """Trả về fingerprint duy nhất cho cache key (theo phản biện #19 + ViSa).

        Format: <served_name>-<8 hex char of model id hash>.
        """
        if self._model_fingerprint is not None:
            return self._model_fingerprint
        try:
            data = await self.list_models()
            model_id = ""
            if (models := data.get("data")) and isinstance(models, list) and models:
                model_id = str(models[0].get("id", ""))
            import blake3 as _blake3

            digest = _blake3.blake3(model_id.encode()).hexdigest()[:8] if model_id else "unknown"
            self._model_fingerprint = f"{self.served_model_name}-{digest}"
        except (httpx.HTTPError, KeyError, ValueError):
            self._model_fingerprint = f"{self.served_model_name}-unknown"
        return self._model_fingerprint


class VllmRegistry:
    """Quản lý các vLLM endpoint trong toàn app.

    Translator và OCR có pool riêng để tránh head-of-line blocking khi 1 model chậm.
    """

    def __init__(self, translator: VllmEndpoint, ocr: VllmEndpoint) -> None:
        self.translator = translator
        self.ocr = ocr

    async def start_all(self) -> None:
        await asyncio.gather(self.translator.start(), self.ocr.start())

    async def stop_all(self) -> None:
        await asyncio.gather(self.translator.stop(), self.ocr.stop())

    async def deep_health_check(self) -> dict[str, Any]:
        translator_health, ocr_health = await asyncio.gather(
            self.translator.deep_health_check(),
            self.ocr.deep_health_check(),
        )
        return {"translator": translator_health, "ocr": ocr_health}
