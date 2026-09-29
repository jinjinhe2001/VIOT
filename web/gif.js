// Minimal GIF89a encoder: indexed frames with one global 256-colour palette.

class ByteWriter {
  constructor() { this.buf = new Uint8Array(1 << 20); this.n = 0; }
  byte(b) {
    if (this.n >= this.buf.length) {
      const nb = new Uint8Array(this.buf.length * 2);
      nb.set(this.buf);
      this.buf = nb;
    }
    this.buf[this.n++] = b;
  }
  bytes(arr) { for (const b of arr) this.byte(b); }
  u16(v) { this.byte(v & 0xff); this.byte((v >> 8) & 0xff); }
  result() { return this.buf.subarray(0, this.n); }
}

// LZW with variable code size, following the timing rules used by omggif.
function lzw(w, indices, minCodeSize = 8) {
  const clear = 1 << minCodeSize, eoi = clear + 1;
  let codeSize = minCodeSize + 1, next = eoi + 1;
  let table = new Map();
  const block = [];
  let cur = 0, curBits = 0;
  const emit = (code) => {
    cur |= code << curBits;
    curBits += codeSize;
    while (curBits >= 8) {
      block.push(cur & 0xff);
      if (block.length === 255) { w.byte(255); w.bytes(block); block.length = 0; }
      cur >>>= 8;
      curBits -= 8;
    }
  };
  w.byte(minCodeSize);
  emit(clear);
  let prefix = indices[0];
  for (let i = 1; i < indices.length; i++) {
    const k = indices[i];
    const key = (prefix << 8) | k;
    const code = table.get(key);
    if (code !== undefined) { prefix = code; continue; }
    emit(prefix);
    if (next === 4096) {
      emit(clear);
      next = eoi + 1;
      codeSize = minCodeSize + 1;
      table = new Map();
    } else {
      if (next >= (1 << codeSize)) codeSize++;
      table.set(key, next++);
    }
    prefix = k;
  }
  emit(prefix);
  emit(eoi);
  if (curBits > 0) { block.push(cur & 0xff); if (block.length === 255) { w.byte(255); w.bytes(block); block.length = 0; } }
  if (block.length) { w.byte(block.length); w.bytes(block); }
  w.byte(0);
}

/** frames: [{ indices: Uint8Array(w*h), delay: centiseconds }], palette: Uint8Array(768). */
export function encodeGIF(frames, width, height, palette) {
  const w = new ByteWriter();
  w.bytes([0x47, 0x49, 0x46, 0x38, 0x39, 0x61]);        // GIF89a
  w.u16(width); w.u16(height);
  w.byte(0xf7); w.byte(0); w.byte(0);                    // global palette, 256 entries
  w.bytes(palette);
  w.bytes([0x21, 0xff, 0x0b]);                           // NETSCAPE2.0 infinite loop
  w.bytes(Array.from('NETSCAPE2.0', (c) => c.charCodeAt(0)));
  w.bytes([0x03, 0x01, 0x00, 0x00, 0x00]);
  for (const f of frames) {
    w.bytes([0x21, 0xf9, 0x04, 0x00]); w.u16(f.delay); w.byte(0); w.byte(0);
    w.byte(0x2c); w.u16(0); w.u16(0); w.u16(width); w.u16(height); w.byte(0);
    lzw(w, f.indices);
  }
  w.byte(0x3b);
  return w.result();
}
