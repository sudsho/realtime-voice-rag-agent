// audio worklet - downsample browser float32 to int16 at 16kHz mono
// posts arraybuffers to the main thread, which forwards to the websocket.

class MicWorklet extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    this.targetSr = opts.targetSr || 16000;
    this.inputSr = sampleRate; // worklet global, set by AudioContext
    this.ratio = this.inputSr / this.targetSr;
    this.acc = 0;
    this.buf = [];
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const channel = input[0];
    // simple decimation - we only sample at output cadence
    for (let i = 0; i < channel.length; i++) {
      this.acc += 1;
      if (this.acc >= this.ratio) {
        this.acc -= this.ratio;
        // clamp + scale to int16
        let s = channel[i];
        if (s > 1) s = 1; else if (s < -1) s = -1;
        this.buf.push(s < 0 ? Math.round(s * 32768) : Math.round(s * 32767));
      }
    }
    // flush every ~20ms (320 samples at 16k)
    if (this.buf.length >= 320) {
      const out = new Int16Array(this.buf);
      this.buf = [];
      this.port.postMessage(out.buffer, [out.buffer]);
    }
    return true;
  }
}

registerProcessor("mic-worklet", MicWorklet);
