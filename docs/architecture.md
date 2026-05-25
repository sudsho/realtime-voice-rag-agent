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

stages overlap where possible:

- audio frames flow to the server as binary WS messages and are buffered
  per session. the browser signals end-of-utterance with a `stop` text
  message (push-to-talk). there is no server-side VAD in the shipped path.
- once `stop` arrives, one faster-whisper transcribe call runs on the full
  buffer, RAG runs on the transcript, then LLM streaming starts.
- LLM tokens accumulate into a sentence buffer. each completed clause is
  handed to TTS. TTS bytes flow back over the same WS as binary frames.
- the browser keeps a small play queue so each TTS chunk is scheduled at
  the previous chunk's tail, with no audible gaps.

no per-stage latency has been measured in this repo. `WARM_MODELS=1` and
ALB sticky cookies exist as a mitigation for cold start, but no timing
figures are recorded or committed.
