// services/image-processing.service.ts
import { Injectable } from '@angular/core';

export interface ProcessedImageResult {
  originalSrc: string;   // base64 data URL
  magnitude: string;
  phase: string;
  real: string;
  imaginary: string;
}

@Injectable({ providedIn: 'root' })
export class ImageProcessingService {

  private readonly baseUrl = 'http://127.0.0.1:8000';

  /**
   * Uploads an image file to the backend.
   * Receives original image + all four FT component images (base64).
   */
  async uploadAndProcess(
    file: File,
    unifyPolicy: 'smallest' | 'largest' | 'fixed',
    keepAspectRatio: boolean,
  ): Promise<ProcessedImageResult> {

    const form = new FormData();
    form.append('image', file, file.name);
    form.append('unify_policy', unifyPolicy);
    form.append('keep_aspect_ratio', String(keepAspectRatio));

    const res = await fetch(`${this.baseUrl}/api/image/upload`, {
      method: 'POST',
      body: form,
    });

    if (!res.ok) {
      throw new Error(`Upload failed: HTTP ${res.status}`);
    }

    const data: {
      original: string;
      magnitude: string;
      phase: string;
      real: string;
      imaginary: string;
    } = await res.json();

    return {
      originalSrc: `data:image/png;base64,${data.original}`,
      magnitude: `data:image/png;base64,${data.magnitude}`,
      phase: `data:image/png;base64,${data.phase}`,
      real: `data:image/png;base64,${data.real}`,
      imaginary: `data:image/png;base64,${data.imaginary}`,
    };
  }
}
