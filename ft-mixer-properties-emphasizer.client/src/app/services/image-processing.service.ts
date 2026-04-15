// services/image-processing.service.ts
import { Injectable } from '@angular/core';

export interface FTComponents {
  original: string;   // base64 data URL
  magnitude: string;
  phase: string;
  real: string;
  imaginary: string;
}

export interface MixerRunResult {
  result_image: string;  // base64 data URL of the mixed spatial result
  magnitude: string;     // FT components of the mixed result
  phase: string;
  real: string;
  imaginary: string;
}

export interface MixerSlotWeights {
  mag_weight: number;
  phase_weight: number;
}

export interface MixerRunOptions {
  component_pair:    'mag-phase' | 'real-imag';
  region_type:       'inner' | 'outer';
  region_size:       number;
  unify_policy:      'smallest' | 'largest' | 'fixed';
  keep_aspect_ratio: boolean;
  weights: [MixerSlotWeights, MixerSlotWeights, MixerSlotWeights, MixerSlotWeights];
}

@Injectable({ providedIn: 'root' })
export class ImageProcessingService {

  private readonly baseUrl = 'http://127.0.0.1:8000';

  // ─────────────────────────────────────────────────────────────────────────
  // Original upload (for properties/emphasizer panel — unchanged)
  // ─────────────────────────────────────────────────────────────────────────

  async uploadAndProcess(
  file:            File,
  unifyPolicy:     'smallest' | 'largest' | 'fixed' = 'smallest',
  keepAspectRatio: boolean = true,
): Promise<FTComponents> {
  const form = new FormData();
  form.append('image',             file, file.name);
  form.append('unify_policy',      unifyPolicy);
  form.append('keep_aspect_ratio', String(keepAspectRatio));

  const res = await fetch(`${this.baseUrl}/api/properties/upload`, {
    method: 'POST',
    body: form,
  });

  if (!res.ok) throw new Error(`Upload failed: HTTP ${res.status}`);

  const data = await res.json();
  return this._jsonToFTComponents(data);
}

  // ─────────────────────────────────────────────────────────────────────────
  // Mixer — upload one slot
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Uploads an image into one of the four mixer slots (0–3).
   * Call this once per image. The backend caches the FFT until
   * this slot is replaced or cleared.
   *
   * Returns the slot's original image + FT components so the UI
   * can display them immediately after upload.
   */
  async mixerUploadSlot(
    slot:         0 | 1 | 2 | 3,
    file:         File,
    mag_weight:   number = 1.0,
    phase_weight: number = 1.0,
  ): Promise<FTComponents> {

    const form = new FormData();
    form.append('slot',         String(slot));
    form.append('image',        file, file.name);
    form.append('mag_weight',   String(mag_weight));
    form.append('phase_weight', String(phase_weight));

    const res = await fetch(`${this.baseUrl}/api/mixer/upload`, {
      method: 'POST',
      body: form,
    });

    if (!res.ok) throw new Error(`Mixer upload failed: HTTP ${res.status}`);

    const data = await res.json();
    return this._jsonToFTComponents(data);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Mixer — run the mix
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Runs the mix across all loaded slots.
   * No images are sent — the backend reuses the cached FFTs.
   * Only weights + mix settings are sent.
   */
  async mixerRun(options: MixerRunOptions): Promise<MixerRunResult> {
    const form = new FormData();

    form.append('component_pair',    options.component_pair);
    form.append('region_type',       options.region_type);
    form.append('region_size',       String(options.region_size));
    form.append('unify_policy',      options.unify_policy);
    form.append('keep_aspect_ratio', String(options.keep_aspect_ratio));

    // Per-slot weight overrides
    options.weights.forEach((w, i) => {
      form.append(`mag_weight_${i}`,   String(w.mag_weight));
      form.append(`phase_weight_${i}`, String(w.phase_weight));
    });

    const res = await fetch(`${this.baseUrl}/api/mixer/run`, {
      method: 'POST',
      body: form,
    });

    if (!res.ok) throw new Error(`Mixer run failed: HTTP ${res.status}`);

    const data = await res.json();

    return {
      result_image: `data:image/jpeg;base64,${data.result_image}`,
      magnitude:    `data:image/jpeg;base64,${data.magnitude}`,
      phase:        `data:image/jpeg;base64,${data.phase}`,
      real:         `data:image/jpeg;base64,${data.real}`,
      imaginary:    `data:image/jpeg;base64,${data.imaginary}`,
    };
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Mixer — clear one slot
  // ─────────────────────────────────────────────────────────────────────────

  async mixerClearSlot(slot: 0 | 1 | 2 | 3): Promise<void> {
    const form = new FormData();
    form.append('slot', String(slot));

    const res = await fetch(`${this.baseUrl}/api/mixer/clear`, {
      method: 'POST',
      body: form,
    });

    if (!res.ok) throw new Error(`Mixer clear failed: HTTP ${res.status}`);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private helpers
  // ─────────────────────────────────────────────────────────────────────────

  private _jsonToFTComponents(data: Record<string, string>): FTComponents {
    return {
      original:  `data:image/jpeg;base64,${data["original"]}`,
      magnitude: `data:image/jpeg;base64,${data["magnitude"]}`,
      phase:     `data:image/jpeg;base64,${data["phase"]}`,
      real:      `data:image/jpeg;base64,${data["real"]}`,
      imaginary: `data:image/jpeg;base64,${data["imaginary"]}`,
    };
  }
}