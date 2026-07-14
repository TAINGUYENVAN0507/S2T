from dataclasses import dataclass

import numpy as np
import torch
import webrtcvad
from silero_vad import load_silero_vad

from app.config import settings
from app.core.errors import VADProcessingError


@dataclass(slots=True)
class SpeechSegment:
    """
    Một đoạn audio được xác định là có tiếng nói.
    """

    start_sample: int
    end_sample: int
    pcm_bytes: bytes

    @property
    def duration_seconds(self) -> float:
        return (self.end_sample - self.start_sample) / settings.SAMPLE_RATE


class TwoStageVAD:
    """
    VAD hai lớp:

    Lớp 1:
        WebRTC VAD lọc nhanh silence rõ ràng.

    Lớp 2:
        Silero VAD kiểm tra lại candidate segment từ WebRTC.
    """

    def __init__(self) -> None:
        self.sample_rate = settings.SAMPLE_RATE

        self.webrtc_vad = webrtcvad.Vad(settings.WEBRTC_VAD_MODE)

        self.silero_model = load_silero_vad(onnx=True)

        self.frame_ms = settings.VAD_FRAME_MS

        if self.frame_ms not in {10, 20, 30}:
            raise ValueError("VAD_FRAME_MS must be one of: 10, 20, 30.")

        self.frame_samples = int(self.sample_rate * self.frame_ms / 1000)

        self.frame_bytes = self.frame_samples * settings.SAMPLE_WIDTH

        self.end_silence_frames = max(
            1,
            int(settings.VAD_END_SILENCE_MS / self.frame_ms),
        )

        self.padding_samples = int(self.sample_rate * settings.VAD_PADDING_MS / 1000)

    def detect(
        self,
        pcm_bytes: bytes,
        sample_rate: int = settings.SAMPLE_RATE,
    ) -> list[SpeechSegment]:
        """
        Nhận PCM16 mono 16 kHz và trả về các đoạn speech.
        """

        try:
            if sample_rate != self.sample_rate:
                raise ValueError(f"Only {self.sample_rate} Hz PCM is supported.")

            if len(pcm_bytes) % settings.SAMPLE_WIDTH != 0:
                raise ValueError("Invalid PCM16 byte length.")

            candidates = self._detect_with_webrtc(
                pcm_bytes=pcm_bytes,
            )

            final_segments = self._verify_with_silero(
                candidates=candidates,
            )

            return final_segments

        except Exception as exc:
            raise VADProcessingError(f"VAD processing failed: {exc}") from exc

    def _detect_with_webrtc(
        self,
        pcm_bytes: bytes,
    ) -> list[SpeechSegment]:
        """
        Dùng WebRTC để tìm candidate speech segments.

        WebRTC yêu cầu input là PCM16 bytes và frame phải là
        10ms, 20ms hoặc 30ms.
        """

        candidates: list[SpeechSegment] = []

        recording = False
        silence_frames = 0

        segment_start_sample: int | None = None
        segment_frames: list[bytes] = []

        total_frames = len(pcm_bytes) // self.frame_bytes

        for frame_index in range(total_frames):
            byte_start = frame_index * self.frame_bytes
            byte_end = byte_start + self.frame_bytes

            frame_bytes = pcm_bytes[byte_start:byte_end]

            is_speech = self.webrtc_vad.is_speech(
                frame_bytes,
                sample_rate=self.sample_rate,
            )

            current_sample = frame_index * self.frame_samples

            if is_speech:
                if not recording:
                    recording = True
                    silence_frames = 0

                    segment_start_sample = max(
                        0,
                        current_sample - self.padding_samples,
                    )

                    segment_frames = []

                    padding_byte_start = segment_start_sample * settings.SAMPLE_WIDTH

                    if padding_byte_start < byte_start:
                        segment_frames.append(pcm_bytes[padding_byte_start:byte_start])

                segment_frames.append(frame_bytes)
                silence_frames = 0

            else:
                if recording:
                    silence_frames += 1
                    segment_frames.append(frame_bytes)

                    if silence_frames >= self.end_silence_frames:
                        if segment_start_sample is None:
                            continue

                        segment_end_sample = current_sample + self.frame_samples

                        candidates.append(
                            SpeechSegment(
                                start_sample=segment_start_sample,
                                end_sample=segment_end_sample,
                                pcm_bytes=b"".join(segment_frames),
                            )
                        )

                        recording = False
                        silence_frames = 0
                        segment_start_sample = None
                        segment_frames = []

        if recording and segment_start_sample is not None and segment_frames:
            segment_end_sample = len(pcm_bytes) // settings.SAMPLE_WIDTH

            candidates.append(
                SpeechSegment(
                    start_sample=segment_start_sample,
                    end_sample=segment_end_sample,
                    pcm_bytes=b"".join(segment_frames),
                )
            )

        return candidates

    def _verify_with_silero(
        self,
        candidates: list[SpeechSegment],
    ) -> list[SpeechSegment]:
        """
        Dùng Silero để xác minh lại candidate segment.

        Cách làm:
        - Convert PCM16 sang float32 [-1, 1].
        - Chia thành frame 512 samples.
        - Tính speech probability trung bình.
        - Giữ segment nếu prob >= SILERO_THRESHOLD.
        """

        final_segments: list[SpeechSegment] = []

        for candidate in candidates:
            waveform = pcm16_to_float32_tensor(candidate.pcm_bytes)

            if waveform.numel() < settings.SILERO_FRAME_SIZE:
                continue

            speech_prob = self._calculate_silero_probability(
                waveform=waveform,
            )

            if speech_prob >= settings.SILERO_THRESHOLD:
                final_segments.append(candidate)

        return final_segments

    def _calculate_silero_probability(
        self,
        waveform: torch.Tensor,
    ) -> float:
        """
        Tính xác suất speech trung bình trên candidate segment.
        """

        frame_size = settings.SILERO_FRAME_SIZE
        probabilities: list[float] = []

        for start in range(0, waveform.numel(), frame_size):
            frame = waveform[start : start + frame_size]

            if frame.numel() < frame_size:
                break

            silero_input = frame.unsqueeze(0)

            with torch.no_grad():
                probability = self.silero_model(
                    silero_input,
                    self.sample_rate,
                ).item()

            probabilities.append(float(probability))

        if not probabilities:
            return 0.0

        return float(np.mean(probabilities))


def pcm16_to_float32_tensor(
    pcm_bytes: bytes,
) -> torch.Tensor:
    """
    Chuyển PCM16 signed little-endian bytes sang tensor float32 [-1, 1].
    """

    pcm_array = np.frombuffer(
        pcm_bytes,
        dtype=np.int16,
    )

    audio_float32 = pcm_array.astype(np.float32) / 32768.0

    return torch.from_numpy(audio_float32)


def join_speech_segments(
    segments: list[SpeechSegment],
    sample_rate: int = settings.SAMPLE_RATE,
    silence_ms: int = 150,
) -> bytes:
    """
    Ghép các đoạn speech thành một đoạn PCM duy nhất.

    Chèn silence ngắn giữa các đoạn để tránh dính câu.
    """

    if not segments:
        return b""

    silence_samples = int(sample_rate * silence_ms / 1000)

    silence_bytes = b"\x00\x00" * silence_samples

    return silence_bytes.join(segment.pcm_bytes for segment in segments)
