"""Chatterbox Multilingual 0.5B engine — async-safe wrapper.

- Load model 1 lần lúc startup (lifespan), warm-up 1 câu để force lazy CUDA kernels.
- `synthesize()` async-safe: bao quanh `asyncio.Semaphore(tts_concurrency)` +
  `asyncio.to_thread` + `asyncio.wait_for`. Mặc định concurrency=1 vì model state share GPU.
- Long text → tách câu (`split_text_for_tts`), generate từng chunk, concat với silence ngăn cách.
- Voice registry load từ `voices/voices.json`, file thiếu → log warning + skip (service vẫn chạy).
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from config import Settings
from models.schemas import VoiceInfo
from services.chunker import split_text_for_tts
from services.text_normalizer import normalize_text
from utils.logging import get_logger

log = get_logger("tts.engine")


@dataclass(frozen=True)
class VoiceProfile:
    id: str
    audio_path: Path | None
    gender: str | None
    language_hint: str | None
    description: str | None


class ChatterboxEngine:
    """Wrap ChatterboxMultilingualTTS với async semaphore + voice registry."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None  # ChatterboxMultilingualTTS — type lazy import
        self._voices: dict[str, VoiceProfile] = {}
        self._semaphore = asyncio.Semaphore(settings.tts_concurrency)
        self._ready = False
        self._sr = 24000

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def sample_rate(self) -> int:
        return self._sr

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def initialize(self) -> None:
        """Load model + voice registry + warm-up. Block lifespan startup."""
        await asyncio.to_thread(self._sync_load)
        self._load_voices()
        await self.warm_up()
        self._ready = True

    def _sync_load(self) -> None:
        # Lazy import — package chatterbox-tts chỉ cần khi service start, không cần ở test.
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # type: ignore[import-not-found]

        log.info(
            "loading_chatterbox",
            model=self._settings.tts_model_id,
            device=self._settings.tts_device,
            dtype=self._settings.tts_dtype,
        )
        self._model = ChatterboxMultilingualTTS.from_pretrained(
            device=self._settings.tts_device,
        )
        sr = getattr(self._model, "sr", None)
        if isinstance(sr, int) and sr > 0:
            self._sr = sr
        log.info("chatterbox_loaded", sr=self._sr)

    def _load_voices(self) -> None:
        manifest_path = Path(self._settings.tts_voices_dir) / self._settings.tts_voices_manifest
        if not manifest_path.exists():
            log.warning("voices_manifest_missing", path=str(manifest_path))
        else:
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                log.error("voices_manifest_invalid", path=str(manifest_path), error=str(exc))
                data = {"voices": []}

            for v in data.get("voices", []):
                voice_id = v.get("id")
                if not voice_id:
                    continue
                audio_path = None
                if v.get("file"):
                    p = Path(self._settings.tts_voices_dir) / v["file"]
                    if not p.exists():
                        log.warning("voice_audio_missing", id=voice_id, path=str(p))
                        continue
                    audio_path = p
                self._voices[voice_id] = VoiceProfile(
                    id=voice_id,
                    audio_path=audio_path,
                    gender=v.get("gender"),
                    language_hint=v.get("language_hint"),
                    description=v.get("description"),
                )

        # Đảm bảo "default" voice luôn có (file=None → dùng default voice của model).
        if "default" not in self._voices:
            self._voices["default"] = VoiceProfile(
                id="default",
                audio_path=None,
                gender=None,
                language_hint=None,
                description="Default voice của Chatterbox (không audio prompt)",
            )

        log.info("voices_loaded", count=len(self._voices), ids=sorted(self._voices.keys()))

    async def warm_up(self) -> None:
        """Generate 1 câu ngắn để force CUDA kernel compile + cache."""
        log.info("warming_up", language=self._settings.tts_warmup_language)
        try:
            wav, _, _ = await self.synthesize(
                text=self._settings.tts_warmup_text,
                language_id=self._settings.tts_warmup_language,
                voice_id=self._settings.tts_warmup_voice_id,
                exaggeration=self._settings.tts_default_exaggeration,
                cfg_weight=self._settings.tts_default_cfg_weight,
                temperature=self._settings.tts_default_temperature,
                seed=42,
            )
            log.info("warm_up_ok", samples=int(wav.shape[-1]))
        except Exception as exc:  # noqa: BLE001
            log.warning("warm_up_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Voice registry queries
    # ------------------------------------------------------------------
    def list_voice_ids(self) -> set[str]:
        return set(self._voices.keys())

    def has_voice(self, voice_id: str) -> bool:
        return voice_id in self._voices

    def list_voices(self) -> list[VoiceInfo]:
        return [
            VoiceInfo(
                id=v.id,
                gender=v.gender,  # type: ignore[arg-type]
                language_hint=v.language_hint,
                description=v.description,
                has_audio_prompt=v.audio_path is not None,
            )
            for v in self._voices.values()
        ]

    # ------------------------------------------------------------------
    # Synthesize
    # ------------------------------------------------------------------
    async def synthesize(
        self,
        *,
        text: str,
        language_id: str,
        voice_id: str,
        exaggeration: float,
        cfg_weight: float,
        temperature: float,
        seed: int | None,
    ) -> tuple[torch.Tensor, int, int]:
        """Generate audio. Trả `(wav_1d_cpu, duration_ms, chunk_count)`.

        Để điều chỉnh tempo, dùng `cfg_weight` (thấp=chậm, cao=sát voice prompt).
        Raises `asyncio.TimeoutError` nếu inference quá `tts_inference_timeout_s`.
        """
        if voice_id not in self._voices:
            raise KeyError(voice_id)

        # Text normalization trước khi chunk: 1000 → one thousand, $50 → 50 dollars, ...
        if self._settings.tts_normalize_numbers:
            text = normalize_text(text, language_id)

        chunks = split_text_for_tts(text, self._settings.tts_chunk_size_chars)
        if not chunks:
            raise ValueError("text rỗng sau khi chunk")

        async with self._semaphore:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    self._sync_generate,
                    chunks,
                    language_id,
                    voice_id,
                    exaggeration,
                    cfg_weight,
                    temperature,
                    seed,
                ),
                timeout=self._settings.tts_inference_timeout_s,
            )

    def _sync_generate(
        self,
        chunks: list[str],
        language_id: str,
        voice_id: str,
        exaggeration: float,
        cfg_weight: float,
        temperature: float,
        seed: int | None,
    ) -> tuple[torch.Tensor, int, int]:
        assert self._model is not None, "Engine chưa initialize"

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        profile = self._voices[voice_id]
        prompt_path = str(profile.audio_path) if profile.audio_path else None

        silence_samples = int(self._sr * self._settings.tts_chunk_silence_ms / 1000)
        silence = torch.zeros(silence_samples, dtype=torch.float32) if silence_samples > 0 else None

        wavs: list[torch.Tensor] = []
        for i, chunk in enumerate(chunks):
            kwargs: dict[str, object] = {
                "text": chunk,
                "language_id": language_id,
                "exaggeration": exaggeration,
                "cfg_weight": cfg_weight,
                "temperature": temperature,
            }
            if prompt_path:
                kwargs["audio_prompt_path"] = prompt_path

            wav = self._model.generate(**kwargs)  # type: ignore[arg-type]
            wav = wav.detach().cpu().squeeze()
            if wav.dim() == 0:
                wav = wav.unsqueeze(0)
            wavs.append(wav)

            if silence is not None and i < len(chunks) - 1:
                wavs.append(silence)

        full = torch.cat(wavs, dim=-1) if len(wavs) > 1 else wavs[0]

        # Auto-trim leading/trailing silence + low-energy noise.
        # Chatterbox hay sinh "long_tail" (noise đuôi sau câu) → tạo cảm giác rè rè.
        # Giữ tối thiểu 0.3s để tránh cắt nhầm phụ âm cuối.
        if self._settings.tts_trim_silence:
            import librosa  # type: ignore[import-not-found]
            arr = full.numpy()
            arr_trimmed, _ = librosa.effects.trim(
                arr, top_db=self._settings.tts_trim_top_db,
            )
            min_samples = int(self._sr * 0.3)
            if len(arr_trimmed) >= min_samples:
                full = torch.from_numpy(arr_trimmed.copy())

        duration_ms = int(full.shape[-1] / self._sr * 1000)
        return full, duration_ms, len(chunks)
