# realtime-voice-rag-agent

Voice-interface RAG demo: FastAPI/WebSocket gateway, faster-whisper transcription, ChromaDB retrieval with an optional cross-encoder rerank, and token-streaming OpenAI responses handed clause-by-clause to a streaming TTS provider.

## Why

Support-style voice bots feel slow because they batch the whole pipeline: record, then transcribe, then retrieve, then generate, then synthesize. This repo hands each completed clause from the LLM stream to TTS as it lands, so the TTS provider can start on the first sentence while the LLM is still emitting the rest. No latency numbers are measured in this checkout.

## Stack

- **STT**: faster-whisper (small.en), batch transcription per user turn (single call once the client sends `stop`)
- **RAG**: ChromaDB dense retrieval, with an optional cross-encoder rerank when `sentence-transformers` is installed
- **LLM**: OpenAI API (`gpt-4o-mini` default) via `AsyncOpenAI`
- **TTS**: OpenAI streaming TTS
- **Transport**: WebSocket between browser and FastAPI
- **Deployment**: AWS ECS Fargate + ALB + CloudFront authored in Terraform (never applied from this repo, no committed state)

## Architecture

```
browser  --WS binary PCM-->  FastAPI WS  --buffer-->  faster-whisper (on stop)
                                                              |
                                                         transcript
                                                              v
                                                       chroma retrieve
                                                              |
                                                            context
                                                              v
                                                       llm stream (token by token)
                                                              |
                                                       clause chunks
                                                              v
                                                       streaming TTS
                                                              |
                                                       audio frames
                                                              v
                                     browser  <--WebSocket--  FastAPI WS
```

Endpointing is client-driven: the browser sends `{"type":"stop"}` when the user releases the mic button, and the server then runs one whisper pass on the buffered utterance. There is no server-side VAD in the serving path.

## Quick start (runs offline)

No API keys, no model downloads, no GPU, no cloud. The offline config
(`configs/offline.yaml`) swaps every external backend for a local, dependency-light
stand-in so the whole STT -> RAG -> LLM -> TTS turn runs anywhere:

- **STT** -> a mock ASR that returns a canned transcript from the buffered audio (no faster-whisper)
- **RAG** -> an in-memory hashing-embedding store over the bundled KB articles (no ChromaDB, no embedding key)
- **LLM** -> a local extractive generator that grounds its answer in the retrieved context (no OpenAI key)
- **TTS** -> a mock synthesizer that emits synthetic PCM16 audio bytes (no OpenAI/XTTS)

```bash
make smoke      # or: python scripts/smoke.py
```

Real output:

```
======================================================================
SMOKE 1: full STT -> RAG -> LLM -> TTS pipeline over one turn
======================================================================
  STT backend : MockStt
  RAG store   : LocalVectorStore (7 KB chunks)
  LLM backend : ExtractiveLlm
  TTS backend : MockTts @ 24000 Hz

  fed 19200 synthetic audio samples (~1.20s)

  transcript  : 'How long do I have to get a refund?'
  stages      : ['transcribing', 'retrieving', 'generating']
  retrieved   :
      - 01-billing.md  (score=0.221)
      - 05-warranty.md  (score=0.190)
      - 04-returns.md  (score=0.181)
  answer      : 'We offer a full refund within 14 days of the original purchase if you have not
                 used more than 25 percent of your monthly quota. Approved claims result in either
                 a replacement, a repair, or a refund, at our option. [1][2]'
  tts frames  : 124 (468720 PCM16 bytes)
  metrics(ms) : {'stt_start_ms': 0.03, 'stt_done_ms': 0.04, 'rag_start_ms': 0.07,
                 'rag_done_ms': 0.34, 'llm_first_token_ms': 0.93, 'tts_first_byte_ms': 1.15,
                 'done_ms': 92.26}

  SMOKE 1 PASSED

======================================================================
SMOKE 2: WebSocket gateway in-process (start -> audio -> stop)
======================================================================
  server msg types : ['ready', 'agent_status', 'transcript', 'agent_status', 'agent_status',
                      'tts_meta', 'answer_delta' x41, 'answer_done']
  tts frames       : 124 (468720 bytes)

  SMOKE 2 PASSED

======================================================================
ALL OFFLINE SMOKES PASSED
======================================================================
```

Smoke 1 drives the pipeline directly over one synthetic-audio turn; smoke 2 drives the
FastAPI WebSocket gateway in-process with a scripted `start` / audio / `stop` sequence.

Tests (all offline, no keys):

```bash
pytest -q
# 63 passed in 0.96s
```

## Run locally (full fidelity)

The full-fidelity path uses real faster-whisper, ChromaDB dense retrieval, the OpenAI API,
and OpenAI streaming TTS. It needs an `OPENAI_API_KEY` and downloads the whisper model on
first run.

```bash
# 1. install
make install

# 2. set env
cp .env.example .env
# fill in OPENAI_API_KEY

# 3. ingest sample knowledge base
make kb-ingest

# 4. boot
make serve
# open http://localhost:8000 and click "talk"
```

The XTTS provider (`tts.provider: xtts`) and a real GPU are optional; XTTS also requires the
`coqui-tts` package, which is not declared in `requirements.txt`. Nothing in the offline path
touches whisper, ChromaDB, OpenAI, XTTS, or AWS.

## Run on AWS

```bash
cd terraform
terraform init && terraform apply
# follow the printed ALB URL; CloudFront URL is in outputs
```

The Terraform stack (VPC, subnets, NAT, ALB with sticky cookies, ECS Fargate service/task, CloudFront, Secrets Manager wiring) is complete but has not been applied from this repo. There is no committed tfstate. ECS `desired_count` is fixed; no autoscaling policy is declared.

## What's in the repo

- `src/stt/` - faster-whisper wrapper (`whisper_stream.py`) plus an offline `mock_stt.py`; `factory.py` picks the backend from `stt.backend`. `vad.py` contains a Silero-VAD wrapper and utterance segmenter that are not wired into the serving path.
- `src/tts/` - OpenAI streaming TTS provider, an offline `mock_tts.py`, and a factory. A stub `xtts.py` is present but its dependency is not declared.
- `src/rag/` - Chroma store plus an offline `local_store.py` (hashing-embedding in-memory store), retriever with optional cross-encoder rerank, ingest CLI, and `factory.py` to pick the backend from `rag.backend`. `bm25.py` and `hybrid.py` implement a sparse index and RRF fusion respectively, unit-tested only, not wired into the pipeline.
- `src/agent/` - pipeline wiring STT -> RAG -> LLM -> TTS as async coroutines; an offline extractive `local_llm.py` and an `llm_factory.py`; sentence buffer for clause-by-clause TTS handoff; barge-in detector class (not instantiated by the serving path).
- `scripts/smoke.py` - offline end-to-end smoke (`make smoke`): runs the full pipeline and the WebSocket gateway in-process with the mock/local backends.
- `src/audio.py` - resample + PCM utils
- `src/ws/server.py` - WebSocket gateway
- `src/api/main.py` - FastAPI server (HTTP health + WS endpoint)
- `frontend/` - minimal HTML + JS for mic capture and audio playback
- `terraform/` - ECS Fargate + ALB + CloudFront (not applied)
- `tests/` - unit tests for audio, chunker, RAG retrieve, sentence buffer, BM25, hybrid retrieval, metrics, config, protocol, TTS factory, VAD utilities, health endpoint
- `configs/` - default + production knobs
- `notebooks/latency_profile.ipynb` - scaffold for parsing structured logs. No executed cells, no committed log data.

## Known limitations

- Cold start is dominated by whisper model load. Default config is `device: cpu` / `compute_type: int8`; no GPU is required or exercised.
- Endpointing in the serving path is a client-side push-to-talk stop message, not automatic VAD. The VAD/utterance-segmenter modules and the barge-in detector are implemented and unit-tested but not wired into `ws/server.py`.
- Hybrid retrieval (BM25 + dense + RRF) is implemented in `src/rag/hybrid.py` and unit-tested. The running pipeline uses dense-only retrieval.
- The Terraform stack is authored but has never been applied from this repo. No live deployment, no CloudWatch metrics, no autoscaling policy.
- No latency benchmarks have been run in this checkout. Any per-stage figures in the docs are budgets, not measurements.
- The cross-encoder reranker requires `sentence-transformers`, which is not pinned in `requirements.txt`; without it the retriever falls back to raw Chroma ordering.

## Ethics

Voice cloning capabilities are intentionally not wired into the demo path. Do not use this system to impersonate real individuals.

## License

MIT
