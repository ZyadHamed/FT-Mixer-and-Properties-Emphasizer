import { Component, ChangeDetectorRef } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DecimalPipe } from '@angular/common';

export type EmphasizerAction =
  | 'shift' | 'stretch' | 'mirror' | 'rotate'
  | 'even'  | 'odd'
  | 'complex_exp' | 'window'
  | 'differentiate' | 'integrate'
  | 'ft_repeat';

export interface EmphasizerParams {
  shiftX: number;
  shiftY: number;
  cyclicShift: boolean;
  flipShift: boolean;
  scaleX: number;
  scaleY: number;
  angle: number;
  expandCanvas: boolean;
  mirrorAxis: 'horizontal' | 'vertical' | 'both';
  duplicateMode: boolean;
  freqU: number;
  freqV: number;
  windowType: 'rectangular' | 'gaussian' | 'hamming' | 'hanning';
  sigma: number;
  windowW: number;
  windowH: number;
  ftRepeat: number;
  ftScenarioType: 'A' | 'B' | 'C';
  chainFT: number;
}

const BASE = 'http://127.0.0.1:8000';

@Component({
  selector: 'app-ft-emphasizer',
  templateUrl: './ft-emphasizer.component.html',
  styleUrls: ['./ft-emphasizer.component.css'],
  imports: [FormsModule, DecimalPipe],
})
export class FtEmphasizerComponent {

  constructor(private cdr: ChangeDetectorRef) {}

  selectedAction: EmphasizerAction = 'shift';
  domain: 'spatial' | 'frequency' = 'spatial';

  params: EmphasizerParams = {
    shiftX: 0, shiftY: 0, cyclicShift: true, flipShift: false,
    scaleX: 1, scaleY: 1,
    angle: 0, expandCanvas: true,
    mirrorAxis: 'horizontal', duplicateMode: false,
    freqU: 0, freqV: 0,
    windowType: 'gaussian', sigma: 30, windowW: 256, windowH: 256,
    ftRepeat: 1, ftScenarioType: 'A',
    chainFT: 0,
  };

  // ─── State ────────────────────────────────────────────────────
  originalLoaded = false;
  resultReady    = false;
  loading        = false;
  resultIsComplex = false;

  private originalFile: File | null = null;

  // Display sources
  originalSrc:  string | null = null;
  resultSrc:    string | null = null;
  ftOrigSrc:    string | null = null;
  ftResultSrc:  string | null = null;

  // Display modes
  spatialOrigMode   = 'image';
  spatialResultMode = 'image';
  ftOrigMode        = 'magnitude';
  ftResultMode      = 'magnitude';

  // Cached blobs for mode-switching without re-fetch
  private ftOrigBlobs:   Record<string, string> = {};
  private ftResultBlobs: Record<string, string> = {};
  private _pendingComplexBlobs: Record<string, string> | null = null;
  private spatialResultBlobs: Record<string, string> = {};


  // ─── UI helpers ───────────────────────────────────────────────
  onActionChange(): void {}

  onFtOrigModeChange(): void {
    this.ftOrigSrc = this.ftOrigBlobs[this.ftOrigMode] ?? null;
  }

  onFtResultModeChange(): void {
    this.ftResultSrc = this.ftResultBlobs[this.ftResultMode] ?? null;
  }

  onSpatialResultModeChange(): void {
  this.resultSrc = this.spatialResultBlobs[this.spatialResultMode] ?? null;
}

  // ─── Image loading ────────────────────────────────────────────
  browseImage(): void {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.onchange = async (e: Event) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;
      this.originalFile   = file;
      this.originalSrc    = await this.fileToDataURL(file);
      this.originalLoaded = true;
      this.resultReady    = false;
      this.ftOrigBlobs    = {};
      this.ftResultBlobs  = {};
      await this.computeFftOfOriginal(file);
      this.cdr.detectChanges();
    };
    input.click();
  }

  private fileToDataURL(file: File): Promise<string> {
    return new Promise(res => {
      const r = new FileReader();
      r.onload = ev => res(ev.target!.result as string);
      r.readAsDataURL(file);
    });
  }

  // ─── FFT of original (auto on load) ──────────────────────────
  private async computeFftOfOriginal(file: File): Promise<void> {
    this.loading = true;
    try {
      const form = new FormData();
      form.append('scenario_type', 'A');
      form.append('image', file);

      const res = await fetch(`${BASE}/fft?n=1`, { method: 'POST', body: form });
      if (!res.ok) throw new Error(`FFT failed: ${res.status}`);

      const parts = await this.parseMultipart(res);
      this.ftOrigBlobs = {
        magnitude: parts['magnitude'],
        phase:     parts['phase'],
        real:      parts['real'],
        imaginary: parts['imaginary'],
      };
      this.ftOrigSrc = this.ftOrigBlobs[this.ftOrigMode];
      this.cdr.detectChanges();
    } catch (err) {
      console.error('computeFftOfOriginal error:', err);
    } finally {
      this.loading = false;
      this.cdr.detectChanges();
    }
  }

  // ─── Main apply ───────────────────────────────────────────────
  async applyAction(): Promise<void> {
    if (!this.originalLoaded || this.loading) return;
    if (this.domain === 'spatial') {
      await this.applyOnSpatial();
    } else {
      await this.applyOnFrequency();
    }
  }

  // ─── SPATIAL PIPELINE ─────────────────────────────────────────
  // 1. Call the spatial operation endpoint with the original file.
  // 2. Display the result image in the spatial output port.
  // 3. Re-fetch the blob, convert to File, send to /fft to populate FT output port.
  private async applyOnSpatial(): Promise<void> {
    if (!this.originalFile) return;
    this.loading = true;
    this.resultReady = false;
    try {

      const spatialBlobUrl = await this.callSpatialOperation(this.originalFile);
      if (this._pendingComplexBlobs) {
        this.spatialResultBlobs = this._pendingComplexBlobs;
        this._pendingComplexBlobs = null;
        this.resultIsComplex = true;
        this.spatialResultMode = 'magnitude'; // default to magnitude for complex
      } else {
        this.spatialResultBlobs = { image: spatialBlobUrl };
        this.resultIsComplex = false;
        this.spatialResultMode = 'image';
      }
      this.resultSrc = this.spatialResultBlobs[this.spatialResultMode] ?? spatialBlobUrl;

      // FT viewport always gets the actual FFT of the magnitude result
      const n = Math.max(1, this.params.chainFT);
      this.ftResultBlobs = await this.callFftOnBlobUrl(spatialBlobUrl, 'A', n);
      this.ftResultSrc   = this.ftResultBlobs[this.ftResultMode];

      this.resultReady = true;
      this.cdr.detectChanges();
    } catch (err) {
      console.error('applyOnSpatial error:', err);
    } finally {
      this.loading = false;
      this.cdr.detectChanges();
    }
  }

  // ─── FREQUENCY PIPELINE ───────────────────────────────────────
  // Sends the original image file to /fft_then_operate.
  // Backend: FFTs → applies operation → IFFTs → returns spatial JPEG + 4 FT JPEGs.
  private async applyOnFrequency(): Promise<void> {
    if (!this.originalFile) return;
    this.loading = true;
    this.resultReady = false;
    try {
      const form = this.buildActionForm(this.originalFile);
      const res  = await fetch(`${BASE}/fft_then_operate`, { method: 'POST', body: form });
      if (!res.ok) throw new Error(`fft_then_operate failed: ${res.status}`);

      const parts = await this.parseMultipart(res);
      this.resultSrc     = parts['spatial'];
      this.resultIsComplex = false;
      this.spatialResultMode = 'image';
      this.ftResultBlobs = {
        magnitude: parts['magnitude'],
        phase:     parts['phase'],
        real:      parts['real'],
        imaginary: parts['imaginary'],
      };
      this.ftResultSrc = this.ftResultBlobs[this.ftResultMode];

      this.resultReady = true;
      this.cdr.detectChanges();
    } catch (err) {
      console.error('applyOnFrequency error:', err);
    } finally {
      this.loading = false;
      this.cdr.detectChanges();
    }
  }

  // ─── SPATIAL OPERATION DISPATCHER ────────────────────────────
  private async callSpatialOperation(file: File): Promise<string> {
    switch (this.selectedAction) {
      case 'shift':         return this.opShift(file);
      case 'stretch':       return this.opStretch(file);
      case 'mirror':        return this.opMirror(file);
      case 'rotate':        return this.opRotate(file);
      case 'even':          return this.opEvenOdd(file, 'even');
      case 'odd':           return this.opEvenOdd(file, 'odd');
      case 'complex_exp':   return this.opComplexExp(file);
      case 'window':        return this.opWindow(file);
      case 'differentiate': return this.opDiff(file);
      case 'integrate':     return this.opIntegrate(file);
      case 'ft_repeat':     return this.opFtRepeat(file);
      default: throw new Error(`Unknown action: ${this.selectedAction}`);
    }
  }

  // ─── INDIVIDUAL SPATIAL OPERATION HANDLERS ───────────────────

  private async opShift(file: File): Promise<string> {
    return this.postAndGetImage(`${BASE}/shiftimage`, this.buildForm(file, {
      shift_x: this.params.shiftX,
      shift_y: this.params.shiftY,
      cyclic:  this.params.cyclicShift,
      flip:    this.params.flipShift,
    }));
  }

  private async opStretch(file: File): Promise<string> {
    return this.postAndGetImage(`${BASE}/stretchimage`, this.buildForm(file, {
      stretch_x: this.params.scaleX,
      stretch_y: this.params.scaleY,
    }));
  }

  private async opMirror(file: File): Promise<string> {
    return this.postAndGetImage(`${BASE}/mirrorimage`, this.buildForm(file, {
      mirror_axis:    this.params.mirrorAxis,
      duplicate_mode: this.params.duplicateMode,
    }));
  }

  private async opRotate(file: File): Promise<string> {
    return this.postAndGetImage(`${BASE}/rotateimage`, this.buildForm(file, {
      angle: this.params.angle,
    }));
  }

  private async opEvenOdd(file: File, type: 'even' | 'odd'): Promise<string> {
    return this.postAndGetImage(`${BASE}/makeimageevenorodd`, this.buildForm(file, {
      symmetry_type: type,
    }));
  }

  private async opComplexExp(file: File): Promise<string> {
    const form = this.buildForm(file, {
      amplitude: 1,
      freq_u:    this.params.freqU,
      freq_v:    this.params.freqV,
    });
    const res = await fetch(`${BASE}/multiplybycomplexexponential`, { method: 'POST', body: form });
    if (!res.ok) throw new Error(`multiplybycomplexexponential failed: ${res.status}`);
    const parts = await this.parseMultipart(res);
    // Stash all 4 display components so applyOnSpatial can pick them up
    this._pendingComplexBlobs = {
      magnitude: parts['magnitude'],
      phase:     parts['phase'],
      real:      parts['real'],
      imaginary: parts['imaginary'],
    };
    return parts['magnitude'];
  }

  private async opWindow(file: File): Promise<string> {
    return this.postAndGetImage(`${BASE}/multiplybywindow`, this.buildForm(file, {
      window_type: this.params.windowType,
      ...(this.params.windowType === 'gaussian'
        ? { sigma_x: this.params.sigma, sigma_y: this.params.sigma }
        : {}),
    }));
  }

  private async opDiff(file: File): Promise<string> {
    return this.postAndGetImage(`${BASE}/differentiateimage`, this.buildForm(file, { axis: 'x' }));
  }

  private async opIntegrate(file: File): Promise<string> {
    return this.postAndGetImage(`${BASE}/integrateimage`, this.buildForm(file, { axis: 'x' }));
  }

  private async opFtRepeat(file: File): Promise<string> {
    const form = this.buildForm(file, { scenario_type: this.params.ftScenarioType });
    const res  = await fetch(`${BASE}/fft?n=${this.params.ftRepeat}`, { method: 'POST', body: form });
    if (!res.ok) throw new Error(`ft_repeat failed: ${res.status}`);
    const parts = await this.parseMultipart(res);
    return parts['magnitude'];
  }

  // ─── FFT ON A BLOB URL ────────────────────────────────────────
  // Re-fetches the local blob, wraps it as a File, sends to /fft.
  private async callFftOnBlobUrl(
    blobUrl: string,
    scenario: 'A' | 'B' | 'C',
    n: number
  ): Promise<Record<string, string>> {
    const blob = await fetch(blobUrl).then(r => r.blob());
    const file = new File([blob], 'result.jpg', { type: 'image/jpeg' });
    const form = new FormData();
    form.append('scenario_type', scenario);
    form.append('image', file);
    const res = await fetch(`${BASE}/fft?n=${n}`, { method: 'POST', body: form });
    if (!res.ok) throw new Error(`FFT on result failed: ${res.status}`);
    const parts = await this.parseMultipart(res);
    return {
      magnitude: parts['magnitude'],
      phase:     parts['phase'],
      real:      parts['real'],
      imaginary: parts['imaginary'],
    };
  }

  // ─── FORM BUILDERS ────────────────────────────────────────────

  private buildForm(
    file: File,
    fields: Record<string, string | number | boolean>
  ): FormData {
    const form = new FormData();
    for (const [k, v] of Object.entries(fields)) form.append(k, String(v));
    form.append('image', file);
    return form;
  }

  private buildActionForm(file: File): FormData {
    const form = new FormData();
    form.append('action',         this.selectedAction);
    form.append('shift_x',        String(this.params.shiftX));
    form.append('shift_y',        String(this.params.shiftY));
    form.append('cyclic',         String(this.params.cyclicShift));
    form.append('flip',           String(this.params.flipShift));
    form.append('stretch_x',      String(this.params.scaleX));
    form.append('stretch_y',      String(this.params.scaleY));
    form.append('angle',          String(this.params.angle));
    form.append('mirror_axis',    this.params.mirrorAxis);
    form.append('duplicate_mode', String(this.params.duplicateMode));
    form.append('amplitude',      '1');
    form.append('freq_u',         String(this.params.freqU));
    form.append('freq_v',         String(this.params.freqV));
    form.append('window_type',    this.params.windowType);
    form.append('sigma_x',        String(this.params.sigma));
    form.append('sigma_y',        String(this.params.sigma));
    form.append('symmetry_type',  this.selectedAction === 'even' ? 'even' : 'odd');
    form.append('axis',           'x');
    form.append('scenario_type',  this.params.ftScenarioType);
    form.append('n',              String(this.params.ftRepeat));
    form.append('image',          file);
    return form;
  }

  // ─── FETCH HELPERS ────────────────────────────────────────────

  /** POST and return a blob URL. Handles plain JPEG and multipart (returns magnitude). */
  private async postAndGetImage(url: string, form: FormData): Promise<string> {
    const res = await fetch(url, { method: 'POST', body: form });
    if (!res.ok) throw new Error(`${url} failed: ${res.status}`);
    const ct = res.headers.get('Content-Type') ?? '';
    if (ct.startsWith('image/')) {
      return URL.createObjectURL(await res.blob());
    }
    const parts = await this.parseMultipart(res);
    return parts['magnitude'];
  }

  // ─── MULTIPART PARSER ─────────────────────────────────────────
  // Only extracts image/* parts — JSON metadata is intentionally ignored.

  private async parseMultipart(res: Response): Promise<Record<string, string>> {
    const ct       = res.headers.get('Content-Type') ?? '';
    const boundary = ct.split('boundary=')[1]?.trim();
    if (!boundary) throw new Error('No multipart boundary in Content-Type');

    const bytes    = new Uint8Array(await res.arrayBuffer());
    const decoder  = new TextDecoder();
    const sepBytes = new TextEncoder().encode(`--${boundary}`);
    const parts: Record<string, string> = {};

    for (const section of this.splitBuffer(bytes, sepBytes)) {
      if (section.length < 4) continue;
      const headerEnd = this.findSequence(section, new Uint8Array([13, 10, 13, 10]));
      if (headerEnd === -1) continue;

      const headerText = decoder.decode(section.slice(0, headerEnd));
      const bodyBytes  = section.slice(headerEnd + 4);
      const name       = headerText.match(/name="([^"]+)"/)?.[1];
      if (!name) continue;

      const partCt = headerText.match(/Content-Type:\s*([^\r\n]+)/i)?.[1]?.trim() ?? '';
      if (!partCt.startsWith('image/')) continue;

      const imgBytes = (bodyBytes[bodyBytes.length - 2] === 13 && bodyBytes[bodyBytes.length - 1] === 10)
        ? bodyBytes.slice(0, -2)
        : bodyBytes;
      parts[name] = URL.createObjectURL(new Blob([imgBytes], { type: partCt }));
    }
    return parts;
  }

  private splitBuffer(buf: Uint8Array, sep: Uint8Array): Uint8Array[] {
    const results: Uint8Array[] = [];
    let start = 0;
    while (start < buf.length) {
      const idx = this.findSequence(buf, sep, start);
      if (idx === -1) break;
      if (idx > start) results.push(buf.slice(start, idx));
      start = idx + sep.length;
      if (buf[start] === 13 && buf[start + 1] === 10) start += 2;
      if (buf[start] === 45 && buf[start + 1] === 45) break;
    }
    return results;
  }

  private findSequence(buf: Uint8Array, seq: Uint8Array, fromIndex = 0): number {
    outer: for (let i = fromIndex; i <= buf.length - seq.length; i++) {
      for (let j = 0; j < seq.length; j++) {
        if (buf[i + j] !== seq[j]) continue outer;
      }
      return i;
    }
    return -1;
  }
}