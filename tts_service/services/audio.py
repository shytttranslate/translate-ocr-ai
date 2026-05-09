"""Audio helpers — encode tensor sang WAV base64.

Dùng `soundfile` (libsndfile) thay vì `torchaudio.save` — torchaudio>=2.11 đổi
backend mặc định sang torchcodec (yêu cầu ffmpeg system + torchcodec package).
soundfile đơn giản hơn, không cần system deps lạ, đã pin trong requirements.
"""
from __future__ import annotations

import base64
import io

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
import torch


def tensor_to_wav_base64(wav: torch.Tensor, sample_rate: int) -> str:
    """Encode audio tensor → WAV PCM 16-bit → base64 string.

    `wav` có thể là [N_samples] (mono) hoặc [channels, N_samples] hoặc [1, N_samples].
    Output WAV mono dùng được trực tiếp với <audio> HTML, ffplay, mpv.
    """
    # Bảo đảm CPU + float32.
    t = wav.detach().cpu()
    if t.dtype != torch.float32:
        t = t.to(torch.float32)

    # Squeeze về 1D (mono). Nếu nhiều kênh → soundfile expect [N, channels].
    if t.dim() == 2:
        # [channels, N] → soundfile cần [N, channels].
        if t.shape[0] in (1, 2) and t.shape[0] < t.shape[1]:
            arr = t.transpose(0, 1).contiguous().numpy()
            if arr.shape[1] == 1:
                arr = arr.squeeze(1)  # mono
        else:
            arr = t.contiguous().numpy()
    else:
        arr = t.contiguous().numpy()

    # Clip để chắc chắn không clip-overflow khi convert float→int16.
    arr = np.clip(arr, -1.0, 1.0)

    buf = io.BytesIO()
    sf.write(buf, arr, sample_rate, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")
