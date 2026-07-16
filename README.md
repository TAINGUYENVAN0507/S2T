# S2T_RT

Speech-to-text API for uploading one or more audio files, normalizing audio with FFmpeg, optionally applying VAD, and transcribing with OpenAI. The public API is exposed through FastAPI and Swagger, while Redis and Celery handle background processing.

## Features

- Upload one or multiple audio files from Swagger.
- Supported formats: `mp3`, `m4a`, `wav`, `flac`, `ogg`, `webm`, `mp4`, `mpeg`, `mpga`.
- Audio normalization to PCM16 mono 16 kHz through FFmpeg.
- Optional two-stage VAD with WebRTC VAD and Silero VAD.
- OpenAI transcription using `OPENAI_MODEL`, default `gpt-4o-transcribe`.
- Upload endpoint waits for the transcription result and returns `files[].text` directly.
- Errors are written to `app/logs/app.log`.

## Architecture

The Docker deployment runs three services:

- `api`: FastAPI application, exposed on port `8004` by default.
- `worker`: Celery worker consuming the `audio_pipeline` queue.
- `redis`: internal Redis broker/backend and temporary audio storage.

Redis is not exposed publicly. Only the API port is published.

## Requirements

- Docker and Docker Compose
- An OpenAI API key
- TCP port `8004` opened on the server firewall/security group

## Environment

Create `.env` from the example:

```bash
cp .env.example .env
```

Edit `.env` and set at least:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-transcribe

API_HOST=0.0.0.0
API_PORT=8004
```

For a server with static IP `14.224.188.206`, keep `API_HOST=0.0.0.0` unless you specifically want Docker to bind only to that address. The API will be reachable at:

```text
http://14.224.188.206:8004/docs
```

## Run With Docker

Build and start all services:

```bash
docker compose up -d --build
```

Check service status:

```bash
docker compose ps
```

View logs:

```bash
docker compose logs -f api
docker compose logs -f worker
```

Application logs are also written to:

```bash
tail -f app/logs/app.log
```

Stop services:

```bash
docker compose down
```

Stop services and remove Redis data:

```bash
docker compose down -v
```

## API Usage

Open Swagger:

```text
http://14.224.188.206:8004/docs
```

### Upload Audio

Endpoint:

```http
POST /audios/
```

Parameters:

- `files`: one or more audio files.
- `language`: transcription language, default `vi`.
- `use_vad`: whether to apply VAD before transcription, default `true`.

The endpoint waits until transcription completes or fails. A successful response includes text per file:

```json
{
  "batch_id": "batch_xxx",
  "status": "completed",
  "total_files": 1,
  "queued_files": 0,
  "processing_files": 0,
  "successful_files": 1,
  "failed_files": 0,
  "files": [
    {
      "file_id": "file_xxx",
      "filename": "audio.wav",
      "status": "success",
      "text": "Transcribed text here.",
      "error": null
    }
  ]
}
```

If a file fails, check `files[].error` and `app/logs/app.log`.

### Queue Status

```http
GET /audios/queue/status
```

Returns current queued and active counts.

## cURL Example

```bash
curl -X POST "http://14.224.188.206:8004/audios/?language=vi&use_vad=false" \
  -F "files=@/path/to/audio.m4a;type=audio/mp4"
```

Multiple files:

```bash
curl -X POST "http://14.224.188.206:8004/audios/?language=vi&use_vad=true" \
  -F "files=@/path/to/audio1.wav" \
  -F "files=@/path/to/audio2.m4a;type=audio/mp4"
```

## Local Development Without Docker

Start Redis separately, then run:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8004 --reload
```

In another terminal:

```bash
celery -A app.workers.celery.celery_app worker -Q audio_pipeline --loglevel=info
```

For local development, set `REDIS_URL=redis://localhost:6379/0` in `.env`.

## Troubleshooting

### Swagger keeps waiting or returns timeout

The Celery worker is probably not running or is not consuming `audio_pipeline`.

```bash
docker compose logs -f worker
```

### Upload succeeds but no text is returned

Check the file-level `error` field in the response and inspect logs:

```bash
tail -f app/logs/app.log
```

### `ModuleNotFoundError: onnxruntime`

Rebuild the Docker image so `onnxruntime` from `requirements.txt` is installed:

```bash
docker compose build --no-cache
docker compose up -d
```

### M4A file fails with empty PCM

The normalizer falls back to a temporary file for `m4a/mp4/mov` containers. If it still fails, check `app/logs/app.log` for FFmpeg stderr.

### Cannot access from server IP

Ensure the service is running and port `8004` is open:

```bash
docker compose ps
sudo ufw allow 8004/tcp
```

Then open:

```text
http://14.224.188.206:8004/docs
```
