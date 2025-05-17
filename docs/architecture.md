# architecture

```
+-----------+      WS       +-------------------+
|  browser  |  <---------> |  FastAPI gateway  |
+-----------+              +---------+---------+
       ^                             |
       |                             v
   <pcm16>                    +-------------+
   speakers                   |  pipeline   |
                              +-------------+
                                |   |   |
                          +-----+   |   +------+
                          v         v          v
                       +----+   +-----+    +------+
                       |STT |   | RAG |    | LLM  |
                       +----+   +-----+    +------+
                          \       |          /
                           \      v         /
                            +-----------+
                            |    TTS    |
                            +-----------+
```

stages overlap on purpose:

- audio frames flow continuously to the server. the server only commits an
  utterance when VAD says the user paused.
- as soon as STT returns a final transcript, RAG runs. typical retrieval is
  under 80 ms when chroma is warm.
- LLM streaming starts immediately and tokens accumulate into a sentence
  buffer. each completed clause is handed to TTS. TTS bytes flow back over
  the same WS as binary frames.
- the browser keeps a small play queue so each TTS chunk is scheduled at
  the previous chunk's tail, with no audible gaps.

target budget per stage on warm cluster, US-East:

| stage              | budget |
|--------------------|--------|
| network up + WS    |  60 ms |
| STT (small.en)     | 250 ms |
| RAG retrieve       |  80 ms |
| LLM first token    | 250 ms |
| TTS first byte     | 200 ms |
| network down       |  60 ms |
| **first audio**    | ~900 ms |

cold start is much worse (~6-8 s for the first turn after a Fargate task
spins up). we mitigate via `WARM_MODELS=1` at startup, and ALB sticky cookies
so the same session sticks to the same warm task.
