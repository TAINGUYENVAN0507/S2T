import queue
import torch
import torchaudio
import numpy as np
import librosa
import asyncio
from transformers import (
    SpeechEncoderDecoderModel,
    AutoFeatureExtractor,
    AutoTokenizer,
    GenerationConfig,
  )
from faster_whisper import WhisperModel
import webrtcvad
from silero_vad import load_silero_vad

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.websockets import WebSocketState
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import logging
import uvicorn

logging.basicConfig(level=logging.DEBUG)  # Set the logging level to DEBUG


CLOSE_SIGNAL = b"CLOSE"  # Signal to stop audio processing
SAMPLE_RATE = 16000  # Required for WebRTC & Silero
WEBRTC_FRAME_SIZE = int(30 * SAMPLE_RATE / 1000)  # 30ms in samples
SILERO_FRAME_SIZE = 512  # Required by Silero VAD
CHANNELS = 1
MIN_SILENCE_FRAMES = int(10)  # 0.2s silence → 6 frames (30ms each)
vad = webrtcvad.Vad(3)  # WebRTC Aggressiveness (0-3)
silero = load_silero_vad(onnx=True)



app = FastAPI()
app_storage_path = "storage"
app.add_middleware(HTTPSRedirectMiddleware)
app.mount("/static", StaticFiles(directory="statics"), name="static")
templates_dir = Path(__file__).parent / "view"
templates = Jinja2Templates(directory=templates_dir)


whisper_model_path = "/home/brainx/.cache/huggingface/hub/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478"
whisper_model = WhisperModel(
  model_size_or_path=whisper_model_path,
  device="cuda",
  compute_type="float16",
)

wav2vec_model_path = '/home/brainx/.cache/huggingface/hub/models--nguyenvulebinh--wav2vec2-bartpho/snapshots/2242a3c94558d5fbede5db5bcf4d0cae1fd6d79c/'
wav2vec_model = SpeechEncoderDecoderModel.from_pretrained(wav2vec_model_path).eval()
wav2vec_feature_extractor = AutoFeatureExtractor.from_pretrained(wav2vec_model_path)
wav2vec_tokenizer = AutoTokenizer.from_pretrained(wav2vec_model_path)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
wav2vec_model = wav2vec_model.to(device)


################################################
################################################
###################UTILS########################
################################################
################################################
def transcribe(audio_buffer: np.ndarray,prefix):
    detected_lang,_ = detect_language_from_buffer(audio_buffer)
    # if detected_lang == "vi":
    #     transcribe_text = transcribe_audio_with_wav2vec(audio_buffer,prefix)
    # else:
    transcribe_text = transcribe_audio_with_whisper(audio_buffer,prefix)
    return transcribe_text

def decode_token_ids(token_ids: list, skip_special_tokens=True) -> str:
    """
    Decodes a sequence of token IDs into text.
    Args:
    token_ids (list): List of token IDs to decode.
    skip_special_tokens (bool): Whether to skip special tokens during decoding.
    Returns:
    str: Decoded text.
    """
    timestamp_begin = wav2vec_tokenizer.vocab_size
    outputs = [token for token in token_ids if token < timestamp_begin]
    return wav2vec_tokenizer.decode(outputs, skip_special_tokens=skip_special_tokens)[1:-1]


import torch
import torchaudio
import numpy as np

def preprocess_audio_from_buffer(audio_buffer: np.ndarray, sample_rate = 16000) -> np.ndarray:
    """
    Preprocess audio from a NumPy buffer: convert to 16kHz mono.
    
    Parameters:
    - audio_buffer (numpy.ndarray): Raw audio data (1D or 2D PCM format).
    - sample_rate (int): Original sample rate of the audio.

    Returns:
    - numpy.ndarray: Processed 16kHz mono audio.
    """

    # Convert to PyTorch tensor
    audio = torch.tensor(audio_buffer, dtype=torch.float32)
    
    # Ensure it's 2D (channels, samples) for processing
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)  # Add channel dimension

    # Convert to mono (average across channels)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    # Resample if not 16kHz
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        audio = resampler(audio)

    return audio.squeeze(0).numpy()  # Return as NumPy array


def remove_silence(audio: np.ndarray, top_db: int = 40) -> np.ndarray:
    """Remove silence from an audio array."""
    non_silent_intervals = librosa.effects.split(audio, top_db=top_db)
    return np.concatenate([audio[start:end] for start, end in non_silent_intervals])

def chunk_audio(audio: np.ndarray, sample_rate: int = 16000, duration: int = 20) -> list:
    """Split audio into 20-second chunks."""
    samples_per_chunk = sample_rate * duration
    return [audio[i:i + samples_per_chunk] for i in range(0, len(audio), samples_per_chunk)]

def transcribe_batch_with_wav2vec(audio_batches: list, prefix) -> str:
    """Transcribe a batch of audio chunks using Wav2Vec2-BARTPho."""
    device = next(wav2vec_model.parameters()).device
    input_values = wav2vec_feature_extractor.pad(
        [{"input_values": batch} for batch in audio_batches],
        padding=True,
        return_tensors="pt",
    )
    generated_ids = wav2vec_model.generate(
        input_values['input_values'].to(device),
        attention_mask=input_values['attention_mask'].to(device),
        decoder_input_ids=wav2vec_tokenizer.batch_encode_plus(
            [prefix] * len(audio_batches), return_tensors="pt"
        )['input_ids'][:, :-1].to(device),
        generation_config=GenerationConfig(
            decoder_start_token_id=wav2vec_tokenizer.bos_token_id
        ),
        max_length=250, num_beams=25, no_repeat_ngram_size=4, num_return_sequences=1,
        early_stopping=True,
    )
    return " ".join([decode_token_ids(seq) for seq in generated_ids])

def transcribe_audio_with_wav2vec(audio_buffer: np.ndarray, prefix) -> str:
    """Process, chunk, and transcribe audio."""
    preprocess_audio_from_buffer(audio_buffer)
    audio = remove_silence(audio_buffer)
    chunks = chunk_audio(audio)
    return " ".join([transcribe_batch_with_wav2vec([chunk],prefix) for chunk in chunks]).strip()


def transcribe_audio_with_whisper( audio_buffer: np.ndarray,prefix, sample_rate: int = 16000,):
        preprocess_audio_from_buffer(audio_buffer)
        audio = remove_silence(audio_buffer)
        chunks = chunk_audio(audio)
        full_transcription = ""
        for chunk in chunks:
          segments, info = whisper_model.transcribe(
              chunk, # Or pass audio_numpy directly here
              vad_filter=True,
              without_timestamps=True,
              language="vi", # Auto-detect language
              initial_prompt=prefix
          )
          for segment in segments :
              full_transcription += segment.text
        return full_transcription.strip()

def detect_language_from_buffer(audio_buffer):
    """
    Detect the language of the audio using Faster Whisper.
    
    Parameters:
    audio_buffer (numpy.ndarray): The raw audio data (PCM format, mono, 16kHz).
    model (WhisperModel): Preloaded Faster Whisper model instance.
    
    Returns:
    tuple: Detected language and probability distribution.
    """
    # Convert to float32 tensor
    audio = torch.tensor(audio_buffer, dtype=torch.float32)
    
    # Ensure it's the correct sample rate (16kHz for Whisper)
    if len(audio.shape) > 1:
        audio = audio.mean(dim=0)  # Convert to mono if stereo
    
    # Run language detection
    segments, info = whisper_model.transcribe(audio.numpy(), language=None)
    
    return info.language, info.language_probability





async def audio_stream_handler(websocket: WebSocket, queue):
    client_config = await websocket.receive_json()
    target_language = client_config.get("target_language", "vi")
    logging.debug(f"Target language selected: {target_language}")
    
    webrtc_buffer = np.array([], dtype=np.float32)
    try:
        while True:
            # Disconnect if the WebSocket state is not connected
            if websocket.client_state != WebSocketState.CONNECTED:
                logging.info("Client disconnected, stopping audio processing")
                break

            # Receive audio data from WebSocket
            audio_data = await websocket.receive_bytes()

            # Handle stop signal
            if audio_data == CLOSE_SIGNAL:
                logging.info("Received CLOSE signal, stopping audio processing")
                break
            
            # logging.debug(f"Received {len(audio_data)} bytes of audio data.")

            # Validate audio data length
            if len(audio_data) % 4 != 0:
                # logging.debug(f"Audio data length {len(audio_data)} is not divisible by 4, skipping.")
                continue
            
            audio_data_np = np.frombuffer(audio_data,dtype=np.float32)
            # logging.debug(f"Converted {len(audio_data)} bytes to np.array of length {len(audio_data_np)}.")

            webrtc_buffer = np.concatenate([webrtc_buffer, audio_data_np])
            # logging.debug(f"Buffer length after concatenation: {len(webrtc_buffer)}.")

            while len(webrtc_buffer) >= WEBRTC_FRAME_SIZE:
                frame = webrtc_buffer[:WEBRTC_FRAME_SIZE]
                await queue.put(frame)  # Send to processing queue
                # logging.debug(f"Queue size: {queue.qsize()} after adding frame.")
                webrtc_buffer = webrtc_buffer[WEBRTC_FRAME_SIZE:]
                # await asyncio.sleep(0.001)
            
    except WebSocketDisconnect:
        logging.debug("Client disconnected")
    except Exception as e:
        logging.error(f"Error during audio processing: {e}")


async def transcribe_handler(websocket,queue):
    recording = False
    buffer = np.array([], dtype=np.float32)
    audio_segments = []
    silence_frames = 0
    transcribe_text=" "
    speech_frame = 0
    prefix=""
    while True:
        audio_data = await queue.get()
        if audio_data is None:
            break  # Exit loop when None is received

        # Process the audio data directly as one frame
        frame = audio_data

        # Convert to int16 for WebRTC VAD
        audio_int16 = (frame * 32767).astype(np.int16).tobytes()
        is_speech_webrtc = vad.is_speech(audio_int16, sample_rate=SAMPLE_RATE)

        if is_speech_webrtc:
            # logging.debug("is speec RTC")
            buffer = np.concatenate([buffer, frame],dtype=np.float32)
            if len(buffer) >= SILERO_FRAME_SIZE:
                silero_input = torch.tensor(buffer[:SILERO_FRAME_SIZE], dtype=torch.float32).unsqueeze(0)
                speech_prob = silero(silero_input, SAMPLE_RATE).item()
                # logging.debug(f"speech prob : {speech_prob:.3f}")
                if speech_prob >= 0.5:
                    # logging.debug("is speec VAD")
                    audio_segments.append(buffer[:SILERO_FRAME_SIZE])
                    recording = True
                    silence_frames = 0
                    speech_frame += 1
                elif silence_frames < 10 and recording == True:
                    silence_frames += 1
                    audio_segments.append(buffer[:SILERO_FRAME_SIZE])
                buffer = buffer[SILERO_FRAME_SIZE:]
        else:
            silence_frames += 1

        if silence_frames >= MIN_SILENCE_FRAMES and speech_frame >= 20 and recording:
            transcribe_text = transcribe(np.concatenate(audio_segments),prefix)
            audio_segments = []
            recording = False
            speech_frame = 0
            silence_frames = 0
        await websocket.send_json({
                    "text": transcribe_text,
                    "language": "vi",
                })
        prefix = transcribe_text
        transcribe_text =" "
                

@app.websocket("/record")
async def handle_audio_stream(websocket: WebSocket):
    """
    WebSocket endpoint to handle real-time audio streaming.
    This accepts an audio stream from the client, processes it, and sends back transcription results.
    Args:
    websocket (WebSocket): WebSocket connection instance.
    """
    logging.debug("New WebSocket connection established for audio streaming.")
    await websocket.accept()
    queue = asyncio.Queue()
    audio_stream_handler_task = asyncio.create_task(audio_stream_handler(websocket,queue))
    transcribe_handler_task = asyncio.create_task(transcribe_handler(websocket,queue))
    try:
        await asyncio.gather(audio_stream_handler_task, transcribe_handler_task)
    except WebSocketDisconnect:
        logging.debug("Client disconnected from WebSocket.")
    except Exception as e:
        logging.error(f"WebSocket error occurred: {e}")
    finally:
        logging.debug("WebSocket connection closed.")

@app.get("/", response_class=HTMLResponse)
async def render_homepage(request: Request):
    """
    Serves the homepage for the application.
    Args:
    request (Request): The HTTP request object.
    Returns:
    HTMLResponse: The rendered HTML page.
    """
    context = {"request": request}
    return templates.TemplateResponse("index.html", context)

if __name__ == "main":
    # Start the FastAPI application with Uvicorn
    uvicorn.run(
    app,
    host="0.0.0.0",
    port=8100,
    ssl_keyfile="/etc/nginx/ssl/connectedbrain.com.vn/privkey.pem",  # Path to SSL private key
    ssl_certfile="/etc/nginx/ssl/connectedbrain.com.vn/fullchain.pem"  # Path to SSL certificate
)
