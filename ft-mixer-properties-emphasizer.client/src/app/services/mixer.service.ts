// services/mixer.service.ts
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

// ── Types ─────────────────────────────────────────────────────────────────

export interface MixerSlotComponents {
  original:  string;   // data URL
  magnitude: string;
  phase:     string;
  real:      string;
  imaginary: string;
}

export interface MixRequest {
  componentPair:    'mag-phase' | 'real-imag';
  regionType:       'inner' | 'outer';
  regionSize:       number;
  unifyPolicy:      'smallest' | 'largest' | 'fixed';
  keepAspectRatio:  boolean;
  weights: [
    { magWeight: number; phaseWeight: number },
    { magWeight: number; phaseWeight: number },
    { magWeight: number; phaseWeight: number },
    { magWeight: number; phaseWeight: number },
  ];
}

export interface MixResultComponents {
  resultSrc:  string;
  magnitude:  string;
  phase:      string;
  real:       string;
  imaginary:  string;
}

export type MixEvent =
  | { type: 'progress'; value: number }
  | { type: 'result' } & MixResultComponents;

// ── Service ───────────────────────────────────────────────────────────────

@Injectable({ providedIn: 'root' })
export class MixerService {

  private readonly baseUrl = 'http://127.0.0.1:8000';

  // Track which slots have been uploaded to the backend
  private readonly _loadedSlots = new Set<number>();

  // ─────────────────────────────────────────────────────────────────────────
  // uploadSlot — call once when the user picks a NEW image for a slot.
  // Backend computes and caches the FFT immediately.
  // Returns the slot's original + FT components for the UI.
  // ─────────────────────────────────────────────────────────────────────────

  uploadSlot(
    slot:        0 | 1 | 2 | 3,
    file:        File,
    magWeight:   number = 1.0,
    phaseWeight: number = 1.0,
  ): Observable<MixerSlotComponents> {
    return new Observable<MixerSlotComponents>(observer => {

      const form = new FormData();
      form.append('slot',         String(slot));
      form.append('image',        file, file.name);
      form.append('mag_weight',   String(magWeight));
      form.append('phase_weight', String(phaseWeight));

      const controller = new AbortController();

      fetch(`${this.baseUrl}/api/mixer/upload`, {
        method: 'POST',
        body:   form,
        signal: controller.signal,
      })
        .then(res => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          return res.json();
        })
        .then((data: {
          original:  string;
          magnitude: string;
          phase:     string;
          real:      string;
          imaginary: string;
        }) => {
          this._loadedSlots.add(slot);
          observer.next({
            original:  `data:image/jpeg;base64,${data.original}`,
            magnitude: `data:image/jpeg;base64,${data.magnitude}`,
            phase:     `data:image/jpeg;base64,${data.phase}`,
            real:      `data:image/jpeg;base64,${data.real}`,
            imaginary: `data:image/jpeg;base64,${data.imaginary}`,
          });
          observer.complete();
        })
        .catch(err => {
          if (err.name !== 'AbortError') observer.error(err);
          else observer.complete();
        });

      return () => controller.abort();
    });
  }

  // ─────────────────────────────────────────────────────────────────────────
  // runMix — no images sent.
  // Backend reuses the cached FFTs from the uploaded slots.
  // Only weights + mix settings travel over the wire.
  // ─────────────────────────────────────────────────────────────────────────

  runMix(req: MixRequest): Observable<MixEvent> {
    return new Observable<MixEvent>(observer => {

      if (this._loadedSlots.size === 0) {
        observer.error(new Error('No images uploaded. Upload at least one slot first.'));
        return;
      }

      const form = new FormData();
      form.append('component_pair',    req.componentPair);
      form.append('region_type',       req.regionType);
      form.append('region_size',       String(req.regionSize));
      form.append('unify_policy',      req.unifyPolicy);
      form.append('keep_aspect_ratio', String(req.keepAspectRatio));

      // Per-slot weight overrides — only weights, no image bytes
      req.weights.forEach((w, i) => {
        form.append(`mag_weight_${i}`,   String(w.magWeight));
        form.append(`phase_weight_${i}`, String(w.phaseWeight));
      });

      // Simulated progress
      let progress = 0;
      const progressInterval = setInterval(() => {
        progress = Math.min(progress + 8, 90);
        observer.next({ type: 'progress', value: progress });
      }, 150);

      const controller = new AbortController();

      fetch(`${this.baseUrl}/api/mixer/run`, {
        method: 'POST',
        body:   form,
        signal: controller.signal,
      })
        .then(res => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          return res.json();
        })
        .then((data: {
          result_image: string;
          magnitude:    string;
          phase:        string;
          real:         string;
          imaginary:    string;
        }) => {
          clearInterval(progressInterval);
          observer.next({ type: 'progress', value: 100 });
          observer.next({
            type:      'result',
            resultSrc: `data:image/jpeg;base64,${data.result_image}`,
            magnitude: `data:image/jpeg;base64,${data.magnitude}`,
            phase:     `data:image/jpeg;base64,${data.phase}`,
            real:      `data:image/jpeg;base64,${data.real}`,
            imaginary: `data:image/jpeg;base64,${data.imaginary}`,
          });
          observer.complete();
        })
        .catch(err => {
          clearInterval(progressInterval);
          if (err.name !== 'AbortError') observer.error(err);
          else observer.complete();
        });

      return () => {
        clearInterval(progressInterval);
        controller.abort();
      };
    });
  }

  // ─────────────────────────────────────────────────────────────────────────
  // clearSlot — removes an image from a backend slot
  // ─────────────────────────────────────────────────────────────────────────

  clearSlot(slot: 0 | 1 | 2 | 3): Observable<void> {
    return new Observable<void>(observer => {

      const form = new FormData();
      form.append('slot', String(slot));

      fetch(`${this.baseUrl}/api/mixer/clear`, {
        method: 'POST',
        body:   form,
      })
        .then(res => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          this._loadedSlots.delete(slot);
          observer.next();
          observer.complete();
        })
        .catch(err => observer.error(err));
    });
  }

  isSlotLoaded(slot: number): boolean {
    return this._loadedSlots.has(slot);
  }

  get loadedSlotCount(): number {
    return this._loadedSlots.size;
  }
}