// browser glue: mic capture, websocket transport, audio playback.

const els = {
  mic: document.getElementById("mic-btn"),
  status: document.getElementById("status"),
  transcript: document.getElementById("transcript"),
  answer: document.getElementById("answer"),
  meters: document.getElementById("meters"),
  metricsList: document.getElementById("metrics"),
  conn: document.getElementById("connstate"),
  reconn: document.getElementById("reconnect"),
};

const state = {
  ws: null,
  audioCtx: null,
  micNode: null,
  recording: false,
  ttsSr: 24000,
  playCursor: 0,
};

function setStatus(s) { els.status.textContent = s; }
function setConn(s) { els.conn.textContent = s; }

async function connect() {
  const url = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws/voice";
  const ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { setConn("connected"); ws.send(JSON.stringify({ type: "start", client_sr: 16000 })); };
  ws.onclose = () => { setConn("disconnected"); state.ws = null; };
  ws.onerror = (e) => { console.error("ws error", e); setStatus("ws error"); };
  ws.onmessage = onMessage;
  state.ws = ws;
}

function onMessage(ev) {
  if (typeof ev.data === "string") {
    let m;
    try { m = JSON.parse(ev.data); } catch { return; }
    handleControl(m);
  } else {
    playPcmChunk(new Int16Array(ev.data));
  }
}

function handleControl(m) {
  switch (m.type) {
    case "ready": setStatus("ready"); break;
    case "transcript":
      els.transcript.textContent = m.text;
      break;
    case "agent_status": setStatus(m.stage); break;
    case "answer_delta":
      els.answer.textContent += m.text;
      break;
    case "tts_meta":
      state.ttsSr = m.sr;
      state.playCursor = state.audioCtx ? state.audioCtx.currentTime : 0;
      break;
    case "answer_done":
      setStatus("idle");
      renderMetrics(m.metrics || {});
      break;
    case "error":
      setStatus("error: " + m.code);
      console.warn("server error", m);
      break;
    default: break;
  }
}

function renderMetrics(metrics) {
  els.metricsList.innerHTML = "";
  Object.entries(metrics).forEach(([k, v]) => {
    const li = document.createElement("li");
    li.textContent = `${k}: ${v}`;
    els.metricsList.appendChild(li);
  });
  els.meters.hidden = false;
}

function playPcmChunk(int16) {
  if (!state.audioCtx) return;
  const ctx = state.audioCtx;
  const f32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) f32[i] = int16[i] / 32768;
  const buf = ctx.createBuffer(1, f32.length, state.ttsSr);
  buf.copyToChannel(f32, 0);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  const now = ctx.currentTime;
  const at = Math.max(state.playCursor, now);
  src.start(at);
  state.playCursor = at + buf.duration;
}

async function startMic() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus("mic API not available");
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
    state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    await state.audioCtx.audioWorklet.addModule("/static/mic-worklet.js");
    const source = state.audioCtx.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(state.audioCtx, "mic-worklet", {
      processorOptions: { targetSr: 16000 },
    });
    node.port.onmessage = (e) => {
      if (state.ws && state.ws.readyState === 1) state.ws.send(e.data);
    };
    source.connect(node);
    state.micNode = node;
    state.recording = true;
    setStatus("listening");
  } catch (err) {
    console.error(err);
    setStatus("mic permission denied");
  }
}

function stopMic() {
  state.recording = false;
  if (state.ws && state.ws.readyState === 1) state.ws.send(JSON.stringify({ type: "stop" }));
  setStatus("processing");
}

els.mic.addEventListener("click", async () => {
  if (!state.ws) await connect();
  if (!state.recording) {
    if (!state.audioCtx) await startMic(); else state.recording = true;
    els.mic.setAttribute("aria-pressed", "true");
    els.answer.textContent = "";
  } else {
    els.mic.setAttribute("aria-pressed", "false");
    stopMic();
  }
});

els.reconn.addEventListener("click", () => { if (!state.ws) connect(); });

connect();
