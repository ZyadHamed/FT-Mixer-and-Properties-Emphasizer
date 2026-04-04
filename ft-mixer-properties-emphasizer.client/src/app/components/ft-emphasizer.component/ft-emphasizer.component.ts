import { Component, ChangeDetectorRef, ElementRef, ViewChild, AfterViewInit, OnDestroy } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DecimalPipe, NgStyle } from '@angular/common';

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
  windowCenterX: number;
  windowCenterY: number;
  ftRepeat: number;
  ftScenarioType: 'A' | 'B' | 'C';
  chainFT: number;
}

const BASE = 'http://127.0.0.1:8000';

@Component({
  selector: 'app-ft-emphasizer',
  templateUrl: './ft-emphasizer.component.html',
  styleUrls: ['./ft-emphasizer.component.css'],
  imports: [FormsModule, DecimalPipe, NgStyle],
})
export class FtEmphasizerComponent implements AfterViewInit, OnDestroy {

  // ViewChild refs for the two input canvases that can show the window overlay
  @ViewChild('spatialOrigCanvas') spatialOrigCanvasRef!: ElementRef<HTMLDivElement>;
  @ViewChild('ftOrigCanvas')      ftOrigCanvasRef!: ElementRef<HTMLDivElement>;

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
    windowCenterX: 0, windowCenterY: 0,
    ftRepeat: 1, ftScenarioType: 'A',
    chainFT: 0,
  };

  // ─── State ────────────────────────────────────────────────────
  originalLoaded = false;
  resultReady    = false;
  loading        = false;
  resultIsComplex = false;
  ftOrigShifted  = true;
  ftResultShifted = true;

  loadingProgress = 0;


  // Natural image dimensions (pixels) — populated once the image loads
  imageNaturalW = 512;
  imageNaturalH = 512;

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

  // ─── Window overlay drag state ────────────────────────────────
  isDraggingWindow = false;
  private dragStartMouseX = 0;
  private dragStartMouseY = 0;
  private dragStartCenterX = 0;
  private dragStartCenterY = 0;
  private boundMouseMove!: (e: MouseEvent) => void;
  private boundMouseUp!:   (e: MouseEvent) => void;

  // ─── Computed helpers for window sliders & overlay ────────────

  /** Half-widths so that the window fully stays within the image. */
  get windowCenterXMin(): number { return -Math.floor((this.imageNaturalW - this.params.windowW) / 2); }
  get windowCenterXMax(): number { return  Math.floor((this.imageNaturalW - this.params.windowW) / 2); }
  get windowCenterYMin(): number { return -Math.floor((this.imageNaturalH - this.params.windowH) / 2); }
  get windowCenterYMax(): number { return  Math.floor((this.imageNaturalH - this.params.windowH) / 2); }

  /** Clamp center values whenever window size changes so they remain valid. */
  clampWindowCenter(): void {
    this.params.windowCenterX = Math.max(this.windowCenterXMin, Math.min(this.windowCenterXMax, this.params.windowCenterX));
    this.params.windowCenterY = Math.max(this.windowCenterYMin, Math.min(this.windowCenterYMax, this.params.windowCenterY));
  }

  /**
   * Returns the overlay rectangle style (percentages relative to the vp-canvas element)
   * for the window overlay on the active input panel.
   */
  get windowOverlayStyle(): Record<string, string> {
    const W = this.imageNaturalW;
    const H = this.imageNaturalH;
    const ww = Math.min(this.params.windowW, W);
    const wh = Math.min(this.params.windowH, H);
    // top-left corner in image pixel coords (image center = W/2, H/2)
    const left = (W / 2 + this.params.windowCenterX - ww / 2);
    const top  = (H / 2 + this.params.windowCenterY - wh / 2);
    return {
      left:   `${(left / W) * 100}%`,
      top:    `${(top  / H) * 100}%`,
      width:  `${(ww   / W) * 100}%`,
      height: `${(wh   / H) * 100}%`,
    };
  }

  /** True when the window overlay should be shown on the spatial-original panel. */
  get showWindowOnSpatial(): boolean {
    return this.selectedAction === 'window' && this.domain === 'spatial' && this.originalLoaded;
  }

  /** True when the window overlay should be shown on the FT-original panel. */
  get showWindowOnFt(): boolean {
    return this.selectedAction === 'window' && this.domain === 'frequency' && this.originalLoaded;
  }

  // ─── Lifecycle ────────────────────────────────────────────────
  ngAfterViewInit(): void {
    this.boundMouseMove = this.onWindowDragMove.bind(this);
    this.boundMouseUp   = this.onWindowDragEnd.bind(this);
  }

  ngOnDestroy(): void {
    document.removeEventListener('mousemove', this.boundMouseMove);
    document.removeEventListener('mouseup',   this.boundMouseUp);
  }

  // ─── Window overlay drag handlers ─────────────────────────────

  onWindowDragStart(event: MouseEvent, canvasEl: HTMLDivElement): void {
    event.preventDefault();
    this.isDraggingWindow   = true;
    this.dragStartMouseX    = event.clientX;
    this.dragStartMouseY    = event.clientY;
    this.dragStartCenterX   = this.params.windowCenterX;
    this.dragStartCenterY   = this.params.windowCenterY;

    // Scale: canvas display width → image pixel width
    const rect = canvasEl.getBoundingClientRect();
    this._dragScaleX = this.imageNaturalW / rect.width;
    this._dragScaleY = this.imageNaturalH / rect.height;

    document.addEventListener('mousemove', this.boundMouseMove);
    document.addEventListener('mouseup',   this.boundMouseUp);
  }

  private _dragScaleX = 1;
  private _dragScaleY = 1;

  private onWindowDragMove(event: MouseEvent): void {
    if (!this.isDraggingWindow) return;
    const dx = (event.clientX - this.dragStartMouseX) * this._dragScaleX;
    const dy = (event.clientY - this.dragStartMouseY) * this._dragScaleY;
    this.params.windowCenterX = Math.max(this.windowCenterXMin, Math.min(this.windowCenterXMax, Math.round(this.dragStartCenterX + dx)));
    this.params.windowCenterY = Math.max(this.windowCenterYMin, Math.min(this.windowCenterYMax, Math.round(this.dragStartCenterY + dy)));
    this.cdr.detectChanges();
  }

  private onWindowDragEnd(_event: MouseEvent): void {
    this.isDraggingWindow = false;
    document.removeEventListener('mousemove', this.boundMouseMove);
    document.removeEventListener('mouseup',   this.boundMouseUp);
  }

  // ─── UI helpers ───────────────────────────────────────────────
  onActionChange(): void {}

  onFtOrigModeChange(): void {
    this.ftOrigSrc = this.ftOrigBlobs[this.ftKey(this.ftOrigMode, this.ftOrigShifted)] ?? null;
  }

  onFtResultModeChange(): void {
    this.ftResultSrc = this.ftResultBlobs[this.ftKey(this.ftResultMode, this.ftResultShifted)] ?? null;
  }

  onSpatialResultModeChange(): void {
    this.resultSrc = this.spatialResultBlobs[this.spatialResultMode] ?? null;
  }

  onFtOrigShiftedToggle(shifted: boolean): void {
    this.ftOrigShifted = shifted;
    this.onFtOrigModeChange();
  }

  onFtResultShiftedToggle(shifted: boolean): void {
    this.ftResultShifted = shifted;
    this.onFtResultModeChange();
  }

  private ftKey(mode: string, shifted: boolean): string {
    return shifted ? mode : `unshifted_${mode}`;
  }

  private fileToGrayscaleDataURL(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width  = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext('2d')!;
      ctx.drawImage(img, 0, 0);
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;
      for (let i = 0; i < data.length; i += 4) {
        const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
        data[i] = data[i + 1] = data[i + 2] = gray;
      }
      ctx.putImageData(imageData, 0, 0);
      resolve(canvas.toDataURL('image/png'));
    };
    img.onerror = reject;
    img.src = URL.createObjectURL(file);
  });
}

private dataURLtoFile(dataURL: string, filename: string): Promise<File> {
  return fetch(dataURL)
    .then(r => r.blob())
    .then(blob => new File([blob], filename, { type: 'image/png' }));
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
      this.originalSrc    = await this.fileToGrayscaleDataURL(file);
      this.originalFile = await this.dataURLtoFile(this.originalSrc, file.name);

      // Read natural dimensions before anything else
      await this.readImageDimensions(this.originalSrc);
      // Reset center to 0,0 when a new image loads
      this.params.windowCenterX = 0;
      this.params.windowCenterY = 0;

      this.originalLoaded = true;
      this.resultReady    = false;
      this.ftOrigBlobs    = {};
      this.ftResultBlobs  = {};
      await this.computeFftOfOriginal(file);
      this.cdr.detectChanges();
    };
    input.click();
  }

  private readImageDimensions(src: string): Promise<void> {
    return new Promise(res => {
      const img = new Image();
      img.onload = () => {
        this.imageNaturalW = img.naturalWidth;
        this.imageNaturalH = img.naturalHeight;
        // Also clamp window size to image dimensions
        this.params.windowW = Math.min(this.params.windowW, this.imageNaturalW);
        this.params.windowH = Math.min(this.params.windowH, this.imageNaturalH);
        res();
      };
      img.src = src;
    });
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
        unshifted_magnitude:  parts['unshifted_magnitude'],
        unshifted_phase:      parts['unshifted_phase'],
        unshifted_real:       parts['unshifted_real'],
        unshifted_imaginary:  parts['unshifted_imaginary'],
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
  private async applyOnSpatial(): Promise<void> {
    if (!this.originalFile) return;
    this.loading = true;
    this.resultReady = false;
    try {
      const spatialBlobUrl = await this.callSpatialOperation(this.originalFile);
      this.loadingProgress = 50;   // step 1 of 2 done
      if (this._pendingComplexBlobs) {
        this.spatialResultBlobs = this._pendingComplexBlobs;
        this._pendingComplexBlobs = null;
        this.resultIsComplex = true;
        this.spatialResultMode = 'magnitude';
      } else {
        this.spatialResultBlobs = { image: spatialBlobUrl };
        this.resultIsComplex = false;
        this.spatialResultMode = 'image';
      }
      this.resultSrc = this.spatialResultBlobs[this.spatialResultMode] ?? spatialBlobUrl;

      // Always compute at least 1 FT of the result (the base pass).
      // chainFT is the number of *extra* passes on top of that, so total = 1 + chainFT.
      const totalFTPasses = 1 + this.params.chainFT;
      this.ftResultBlobs = await this.callFftOnBlobUrl(spatialBlobUrl, 'A', totalFTPasses);
      this.loadingProgress = 100;  // step 2 of 2 done
      // If even FT passes produced a spatial image, show IT as the spatial result
      if (this.ftResultBlobs['spatial_passthrough']) {
        this.resultSrc = this.ftResultBlobs['spatial_passthrough'];
        this.spatialResultBlobs['image'] = this.ftResultBlobs['spatial_passthrough'];
      }

      this.ftResultSrc = this.ftResultBlobs[this.ftKey(this.ftResultMode, this.ftResultShifted)];

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
  private async applyOnFrequency(): Promise<void> {
    if (!this.originalFile) return;
    this.loading = true;
    this.resultReady = false;
    try {
      const form = this.buildActionForm(this.originalFile);
      const res  = await fetch(`${BASE}/fft_then_operate`, { method: 'POST', body: form });
      if (!res.ok) throw new Error(`fft_then_operate failed: ${res.status}`);
      this.loadingProgress = 80;   // network done, parsing remains

      const parts = await this.parseMultipart(res);
      this.loadingProgress = 100;
      this.resultSrc     = parts['spatial'];
      this.resultIsComplex = false;
      this.spatialResultMode = 'image';
      this.ftResultBlobs = {
        magnitude: parts['magnitude'],
        phase:     parts['phase'],
        real:      parts['real'],
        imaginary: parts['imaginary'],
        unshifted_magnitude: parts['unshifted_magnitude'],
        unshifted_phase:     parts['unshifted_phase'],
        unshifted_real:      parts['unshifted_real'],
        unshifted_imaginary: parts['unshifted_imaginary'],
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
    this._pendingComplexBlobs = {
      magnitude: parts['magnitude'],
      phase:     parts['phase'],
      real:      parts['real'],
      imaginary: parts['imaginary'],
    };
    return parts['magnitude'];
  }

  private async opWindow(file: File): Promise<string> {
    // Backend uses absolute pixel coords (top-left origin):
    //   start_x = center_x - window_width//2  must be >= 0
    // Frontend stores an offset from image center, so convert:
    //   abs_center = imageDim/2 + offset
    const absCenterX = Math.round(this.imageNaturalW / 2 + this.params.windowCenterX);
    const absCenterY = Math.round(this.imageNaturalH / 2 + this.params.windowCenterY);
    const fields: Record<string, string | number | boolean> = {
      window_type:   this.params.windowType,
      window_width:  this.params.windowW,
      window_height: this.params.windowH,
      center_x:      absCenterX,
      center_y:      absCenterY,
    };
    if (this.params.windowType === 'gaussian') {
      fields['sigma_x'] = this.params.sigma;
      fields['sigma_y'] = this.params.sigma;
    }
    return this.postAndGetImage(`${BASE}/multiplybywindow`, this.buildForm(file, fields));
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

  const ct = res.headers.get('Content-Type') ?? '';

  if (ct.startsWith('image/')) {
    // Even passes → backend returned a real spatial image (e.g. FT²=flip).
    // Compute 1 FFT of IT for the frequency display panels.
    const spatialBlob = await res.blob();
    const spatialUrl  = URL.createObjectURL(spatialBlob);

    const extraFile = new File([spatialBlob], 'spatial.jpg', { type: 'image/jpeg' });
    const extraForm = new FormData();
    extraForm.append('scenario_type', 'A');
    extraForm.append('image', extraFile);
    const extraRes = await fetch(`${BASE}/fft?n=1`, { method: 'POST', body: extraForm });
    if (!extraRes.ok) throw new Error(`Extra FFT pass failed: ${extraRes.status}`);

    const parts = await this.parseMultipart(extraRes);
    return {
      spatial_passthrough: spatialUrl,  // kept so caller can detect this case
      magnitude:           parts['magnitude'],
      phase:               parts['phase'],
      real:                parts['real'],
      imaginary:           parts['imaginary'],
      unshifted_magnitude: parts['unshifted_magnitude'],
      unshifted_phase:     parts['unshifted_phase'],
      unshifted_real:      parts['unshifted_real'],
      unshifted_imaginary: parts['unshifted_imaginary'],
    };
  }

  const parts = await this.parseMultipart(res);
  return {
    magnitude:           parts['magnitude'],
    phase:               parts['phase'],
    real:                parts['real'],
    imaginary:           parts['imaginary'],
    unshifted_magnitude: parts['unshifted_magnitude'],
    unshifted_phase:     parts['unshifted_phase'],
    unshifted_real:      parts['unshifted_real'],
    unshifted_imaginary: parts['unshifted_imaginary'],
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
    form.append('window_width',   String(this.params.windowW));
    form.append('window_height',  String(this.params.windowH));
    form.append('center_x',       String(Math.round(this.imageNaturalW / 2 + this.params.windowCenterX)));
    form.append('center_y',       String(Math.round(this.imageNaturalH / 2 + this.params.windowCenterY)));
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