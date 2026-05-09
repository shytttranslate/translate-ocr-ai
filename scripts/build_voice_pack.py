#!/usr/bin/env python3
"""Build/refresh voice pack 6 giọng (3 nam + 3 nữ) từ LibriTTS-R.

V1 ship `tts_service/voices/{voice_id}.wav` + `voices.json` đã commit vào repo —
script này CHỈ cần chạy nếu muốn:
- Regen voice pack từ scratch (vd anh muốn đổi speaker)
- Verify Chatterbox vẫn clone OK với 6 speaker hiện tại
- Tìm speaker mới có F0 phân biệt rõ hơn

Output: ghi đè vào `tts_service/voices/`. Sau đó restart vbk-tts.

Yêu cầu:
- Chạy trong .venv-tts (có datasets, librosa, soundfile, torchcodec)
- HF_TOKEN trong env
- vbk-tts đang RUNNING (để verify clone — bỏ qua bằng `--no-verify`)

Usage:
    .venv-tts/bin/python scripts/build_voice_pack.py [--no-verify] [--candidates N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VOICES_DIR = ROOT / "tts_service" / "voices"
TEST_TEXT = "The quick brown fox jumps over the lazy dog."
TTS_URL = "http://127.0.0.1:9004/v1/tts"

# Speaker preset đã verify clone OK với Chatterbox 0.5B Multilingual.
# F0 do librosa.pyin đo trên prompt audio (24kHz mono, trimmed silence).
PRESETS = [
    # voice_id,         spk_id, gender,   F0_Hz, label
    ("male_deep",       "5536", "male",   96.2,  "Bass — giọng nam trầm sâu"),
    ("male_warm",       "3170", "male",   141.2, "Baritone — giọng nam ấm trung"),
    ("male_bright",     "174",  "male",   156.8, "Tenor — giọng nam sáng cao"),
    ("female_warm",     "8842", "female", 186.4, "Alto — giọng nữ ấm thấp"),
    ("female_clear",    "6313", "female", 219.5, "Mezzo — giọng nữ trong trung"),
    ("female_bright",   "2035", "female", 245.0, "Soprano — giọng nữ sáng cao"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-verify", action="store_true",
                   help="Skip step verify Chatterbox clone (không cần vbk-tts running)")
    p.add_argument("--candidates", type=int, default=30,
                   help="Số candidate đầu tiên stream từ LibriTTS-R (mặc định 30, đủ tìm 6 preset)")
    return p.parse_args()


def stream_libritts(target_speakers: set[str], max_candidates: int) -> dict[str, dict]:
    """Stream LibriTTS-R/clean dev.clean, trả {speaker_id: {audio, sr, ...}}."""
    import numpy as np
    import librosa
    from datasets import load_dataset

    print(f"Streaming LibriTTS-R/clean dev.clean (target {len(target_speakers)} speakers)...")
    ds = load_dataset("blabble-io/libritts_r", "clean", split="dev.clean", streaming=True)

    found: dict[str, dict] = {}
    visited = 0
    for sample in ds:
        visited += 1
        sid = str(sample.get("speaker_id"))
        if sid not in target_speakers or sid in found:
            continue

        audio = sample["audio"]
        try:
            if hasattr(audio, "get_all_samples"):
                s = audio.get_all_samples()
                arr = s.data.numpy().squeeze()
                sr = s.sample_rate
            else:
                arr = audio["array"]
                sr = audio["sampling_rate"]
        except Exception as exc:
            print(f"  spk {sid}: decode FAIL {exc}")
            continue

        duration = len(arr) / sr
        if duration < 5.0 or duration > 14.0:
            continue

        if sr != 24000:
            arr = librosa.resample(arr.astype(np.float32), orig_sr=sr, target_sr=24000)
        else:
            arr = arr.astype(np.float32)
        arr, _ = librosa.effects.trim(arr, top_db=30)
        if len(arr) > 24000 * 12:
            arr = arr[:24000 * 12]

        # F0 verification (sanity check vs PRESETS)
        try:
            f0, voiced, _ = librosa.pyin(arr, fmin=70, fmax=400, sr=24000, frame_length=2048)
            mask = voiced & ~np.isnan(f0)
            f0_clean = f0[mask]
            f0_mean = float(np.mean(f0_clean)) if len(f0_clean) >= 50 else None
        except Exception:
            f0_mean = None

        found[sid] = {"audio": arr, "sr": 24000, "f0_mean": f0_mean, "duration": len(arr) / 24000}
        print(f"  spk {sid:>5} dur={len(arr)/24000:.1f}s F0={f0_mean:.1f}Hz" if f0_mean else f"  spk {sid:>5}")

        if len(found) >= len(target_speakers) or visited > max_candidates * 5:
            break

    missing = target_speakers - set(found.keys())
    if missing:
        print(f"\n[WARN] không tìm thấy speakers: {sorted(missing)} (sau {visited} samples). LibriTTS-R có thể đã đổi data.")
    return found


def save_voices(found: dict[str, dict]) -> list[dict]:
    import soundfile as sf
    VOICES_DIR.mkdir(parents=True, exist_ok=True)

    # Cleanup file .wav cũ trong voices/
    for old in VOICES_DIR.glob("*.wav"):
        old.unlink()
        print(f"  removed {old.name}")

    manifest_voices = [
        {
            "id": "default",
            "file": None,
            "gender": None,
            "language_hint": None,
            "description": "Default voice của Chatterbox Multilingual (không audio prompt)",
        }
    ]

    for vid, sid, gender, f0_expected, label in PRESETS:
        if sid not in found:
            print(f"  [SKIP] {vid}: không tìm thấy speaker {sid}")
            continue
        info = found[sid]
        out_path = VOICES_DIR / f"{vid}.wav"
        sf.write(str(out_path), info["audio"], info["sr"], subtype="PCM_16")
        print(f"  saved {out_path.name} ({info['duration']:.1f}s, F0~{info['f0_mean'] or 0:.0f}Hz)")
        manifest_voices.append({
            "id": vid,
            "file": f"{vid}.wav",
            "gender": gender,
            "language_hint": "en",
            "description": f"{label} (LibriTTS-R speaker {sid}, F0~{f0_expected:.0f}Hz)",
        })
    return manifest_voices


def write_manifest(voices: list[dict]) -> None:
    manifest = {
        "$schema_note": "Manifest preset voice cho TTS service. id phải khớp regex ^[a-zA-Z0-9_-]+$. file=null → dùng default voice của Chatterbox.",
        "_source": "LibriTTS-R (Google, CC-BY-4.0) qua HF dataset blabble-io/libritts_r — 6 speakers chọn theo F0 spread.",
        "voices": voices,
    }
    out = VOICES_DIR / "voices.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  wrote {out} ({len(voices)} voices)")


def verify_clone() -> int:
    """Gọi /v1/tts với mỗi voice, verify MD5 != default. Trả số voice fallback."""
    import requests
    print("\nBaseline default synthesize...")
    r = requests.post(TTS_URL, json={"text": TEST_TEXT, "language_id": "en", "voice_id": "default", "seed": 42}, timeout=60)
    r.raise_for_status()
    default_md5 = hashlib.md5(r.json()["audio_base64"].encode()).hexdigest()
    print(f"  default MD5: {default_md5}")

    failed = 0
    for vid, sid, *_ in PRESETS:
        try:
            r = requests.post(TTS_URL, json={"text": TEST_TEXT, "language_id": "en", "voice_id": vid, "seed": 42}, timeout=60)
            r.raise_for_status()
            md5 = hashlib.md5(r.json()["audio_base64"].encode()).hexdigest()
            ok = md5 != default_md5
            print(f"  {vid:<14}: {md5[:12]}... {'OK' if ok else 'FALLBACK ✗'}")
            if not ok:
                failed += 1
        except Exception as exc:
            print(f"  {vid}: ERROR {exc}")
            failed += 1
    return failed


def main() -> int:
    args = parse_args()

    # Stream LibriTTS-R cho chính xác các speaker em đã verify (target_speakers từ PRESETS)
    target_speakers = {sid for _, sid, *_ in PRESETS}
    found = stream_libritts(target_speakers, args.candidates)

    print("\nSaving voice prompts to tts_service/voices/...")
    voices = save_voices(found)

    print("\nWriting manifest...")
    write_manifest(voices)

    if args.no_verify:
        print("\n--no-verify: skip Chatterbox verification.")
        return 0

    print("\nWaiting for vbk-tts to reload manifest (em không tự restart — anh chạy supervisorctl restart vbk-tts trước verify)...")
    print("Hoặc anh có thể chạy lại với --no-verify nếu chỉ build pack.")

    print("\nGiả định vbk-tts đã restart. Verify clone:")
    failed = verify_clone()
    if failed:
        print(f"\n[WARN] {failed}/{len(PRESETS)} voice fallback default. Cần đổi PRESETS hoặc check audio quality.")
        return 2
    print(f"\nAll {len(PRESETS)} voices clone OK ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
