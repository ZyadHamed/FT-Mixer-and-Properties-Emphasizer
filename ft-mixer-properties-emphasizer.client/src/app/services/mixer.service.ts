// services/mixer.service.ts
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

// ── Types ─────────────────────────────────────────────────────────────────
export interface MixImagePayload {
  file: File | null;
  magWeight: number;
  phaseWeight: number;
}

export interface MixRequest {
  images: MixImagePayload[];
  componentPair: 'mag-phase' | 'real-imag';
  regionType: 'inner' | 'outer';
  regionSize: number;
  unifyPolicy: 'smallest' | 'largest' | 'fixed';
  keepAspectRatio: boolean;
}

export interface MixResultComponents {
  resultSrc: string;   // data URL of the spatial result image
  magnitude: string;   // data URL of FT magnitude
  phase: string;   // data URL of FT phase
  real: string;   // data URL of FT real
  imaginary: string;   // data URL of FT imaginary
}

export type MixEvent =
  | { type: 'progress'; value: number }
  | { type: 'result' } & MixResultComponents;

// ── Service ───────────────────────────────────────────────────────────────
@Injectable({ providedIn: 'root' })
export class MixerService {

  private readonly baseUrl = 'http://127.0.0.1:5000';

  runMix(req: MixRequest): Observable<MixEvent> {
    return new Observable<MixEvent>(observer => {

      // Build FormData
      const form = new FormData();
      req.images.forEach((img, i) => {
        if (img.file) form.append(`image_${i}`, img.file, img.file.name);
        form.append(`mag_weight_${i}`, String(img.magWeight));
        form.append(`phase_weight_${i}`, String(img.phaseWeight));
      });
      form.append('component_pair', req.componentPair);
      form.append('region_type', req.regionType);
      form.append('region_size', String(req.regionSize));
      form.append('unify_policy', req.unifyPolicy);
      form.append('keep_aspect_ratio', String(req.keepAspectRatio));

      // Simulated progress
      let progress = 0;
      const progressInterval = setInterval(() => {
        progress = Math.min(progress + 8, 90);
        observer.next({ type: 'progress', value: progress });
      }, 150);

      const controller = new AbortController();

      fetch(`${this.baseUrl}/api/mixer/run`, {
        method: 'POST',
        body: form,
        signal: controller.signal,
      })
        .then(res => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          return res.json();
        })
        .then((data: {
          result_image: string;
          magnitude: string;
          phase: string;
          real: string;
          imaginary: string;
        }) => {
          clearInterval(progressInterval);
          observer.next({ type: 'progress', value: 100 });
          observer.next({
            type: 'result',
            resultSrc: `data:image/jpeg;base64,${data.result_image}`,
            magnitude: `data:image/jpeg;base64,${data.magnitude}`,
            phase: `data:image/jpeg;base64,${data.phase}`,
            real: `data:image/jpeg;base64,${data.real}`,
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
}
