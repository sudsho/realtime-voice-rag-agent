# realtime-voice-rag-agent

low-latency voice agent for customer support. mic audio in, voice answer out,
RAG over a knowledge base in the middle. target round-trip under 1s on a warm
pipeline.

## why

most voice support bots feel slow because they batch the whole pipeline:
record-then-transcribe, then-retrieve, then-generate, then-synthesize. each
stage waits for the previous to fully finish. this repo does it streaming end
to end so the user starts hearing the response while the LLM is still drafting
the rest.

## stack (planned)

- STT: faster-whisper (small.en) with VAD-driven chunking
- RAG: chroma + small reranker
- LLM: openai api or local llama.cpp (config-switch)
- TTS: openai tts streaming, fallback to coqui XTTS-v2
- transport: WebSocket between browser and FastAPI

## status

scaffolding. nothing runs yet.
