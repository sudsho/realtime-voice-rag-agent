# realtime-voice-rag-agent

Low-latency voice agent for customer support. Mic audio in, voice answer out, RAG over a knowledge base in the middle. Target round-trip under 1s on a warm pipeline.

## Why

Most voice support bots feel slow because they batch the whole pipeline: record-then-transcribe, then-retrieve, then-generate, then-synthesize. Each stage waits for the previous to fully finish. This repo does it streaming end-to-end so the user starts hearing the response while the LLM is still drafting the rest.

## Stack

- **STT**: faster-whisper (small.en) with VAD-driven chunking via pyannote-audio
- **RAG**: ChromaDB vector store + small reranker over a customer-support knowledge base
- **LLM**: OpenAI API (`gpt-4o-mini` default) or local llama.cpp (config-switchable)
- **TTS**: OpenAI streaming TTS, fallback to Coqui XTTS-v2
- **Transport**: WebSocket between browser and FastAPI
- **Deployment**: AWS ECS Fargate + ALB + CloudFront via Terraform

## Architecture

```
browser  --WebRTC mic-->  FastAPI WS  --frames-->  VAD  --chunks-->  whisper-stream
                                                                            |
                                                                       transcript
                                                                            v
                                                                     chroma retrieve
                                                                            |
                                                                          context
                                                                            v
                                                                       llm stream (token by token)
                                                                            |
                                                                       text chunks
                                                                            v
                                                                       streaming TTS
                                                                            |
                                                                       audio frames
                                                                            v
                                                       browser  <--WebSocket--  FastAPI WS
```

## Run locally

```bash
# 1. install
make install

# 2. set env
cp .env.example .env
# fill in OPENAI_API_KEY (or LLAMA_CPP_MODEL_PATH)

# 3. ingest sample knowledge base
make ingest

# 4. boot
make serve
# open http://localhost:8000 and click "talk"
```

## Run on AWS

```bash
cd terraform
terraform init && terraform apply
# follow the printed ALB URL; CloudFront URL is in outputs
```

## What's in the repo

- `src/stt/` - whisper streaming wrapper with VAD
- `src/tts/` - XTTS/OpenAI TTS streaming
- `src/rag/` - Chroma indexer + reranker
- `src/agent/` - pipeline wiring STT → RAG → LLM → TTS as async generators
- `src/audio.py` - resample + chunking + PCM utils
- `src/ws/server.py` - WebSocket gateway
- `src/api/main.py` - FastAPI server (HTTP health + WS endpoint)
- `frontend/` - minimal HTML + JS for mic capture and audio playback
- `terraform/` - ECS Fargate + ALB + CloudFront
- `tests/` - unit (audio chunking, RAG retrieve) + integration (WS round-trip with mock LLM)
- `configs/` - default + production knobs
- `notebooks/latency_profile.ipynb` - per-stage latency breakdown

## Known limitations

- Cold start adds ~3s while whisper + XTTS load into GPU memory. The "<1s" target is on a warm pipeline only.
- Without an OpenAI API key, the LLM fallback to llama.cpp expects a local GGUF in `models/` and runs noticeably slower on CPU.

## Ethics

Voice cloning capabilities (XTTS) are intentionally limited to a fixed default voice. The fallback voice cloner is not wired into the demo path. Do not use this system to impersonate real individuals.

## License

MIT
