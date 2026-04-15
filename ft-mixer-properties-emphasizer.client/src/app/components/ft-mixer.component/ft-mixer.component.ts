// ft-mixer.component.ts
import { Component, OnDestroy } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DecimalPipe } from '@angular/common';
import { firstValueFrom } from 'rxjs';

import { Subject, takeUntil } from 'rxjs';
import { ImageViewportComponent, ImageViewportData, FtComponent } from '../image-viewport/image-viewport.component';
import { MixerService, MixRequest } from '../../services/mixer.service';
import { ImageProcessingService } from '../../services/image-processing.service';

export type UnifyPolicy = 'smallest' | 'largest' | 'fixed';
export type ComponentPair = 'mag-phase' | 'real-imag';
export type RegionType = 'inner' | 'outer';

export interface ImageSlot {
  file: File | null;
  viewportData: ImageViewportData | null;
  magWeight: number;
  phaseWeight: number;
  color: string;
  activeComponent: FtComponent;
}

export interface OutputSlot {
  viewportData: ImageViewportData | null;
  active: boolean;
}

@Component({
  selector: 'app-ft-mixer',
  standalone: true,
  templateUrl: './ft-mixer.component.html',
  styleUrls: ['./ft-mixer.component.css'],
  imports: [FormsModule, DecimalPipe, ImageViewportComponent],
})
export class FtMixerComponent implements OnDestroy {

  images: ImageSlot[] = [
    this.makeSlot('#4a9eff'),
    this.makeSlot('#3ecf8e'),
    this.makeSlot('#f7c948'),
    this.makeSlot('#9b8dff'),
  ];

  outputs: OutputSlot[] = [
    { viewportData: null, active: true },
    { viewportData: null, active: false },
  ];

  unifyPolicy: UnifyPolicy = 'smallest';
  keepAspectRatio: boolean = true;
  outputTarget: 0 | 1 = 0;
  componentPair: ComponentPair = 'mag-phase';
  regionType: RegionType = 'inner';
  regionSize: number = 40;

  isMixing = false;
  mixProgress = 0;

  private cancel$ = new Subject<void>();
  private destroy$ = new Subject<void>();

  constructor(
    private mixerService: MixerService,
    private imageProcessingService: ImageProcessingService,
  ) { }

  private makeSlot(color: string): ImageSlot {
    return {
      file: null,
      viewportData: null,
      // Start all 4 at equal share: 1/4 = 0.25
      magWeight: 0.25,
      phaseWeight: 0.25,
      color,
      activeComponent: 'image',
    };
  }

  // ── Normalized weight helpers ──────────────────────────────────────────

  /**
   * Called when the user drags slider for image[index].magWeight.
   * The raw slider value (0–1) is treated as the *raw* weight for that slot.
   * We then normalize all slots so sum = 1, preserving relative ratios for others.
   */
  onMagWeightChange(index: number, rawValue: number): void {
    this.images[index].magWeight = rawValue;
    this._normalizeWeights('mag');
  }

  onPhaseWeightChange(index: number, rawValue: number): void {
    this.images[index].phaseWeight = rawValue;
    this._normalizeWeights('phase');
  }

  private _normalizeWeights(type: 'mag' | 'phase'): void {
    const total = this.images.reduce(
      (sum, img) => sum + (type === 'mag' ? img.magWeight : img.phaseWeight),
      0,
    );

    if (total === 0) {
      // Edge case: all zero → distribute equally
      const eq = 1 / this.images.length;
      this.images.forEach(img => {
        if (type === 'mag') img.magWeight = eq;
        else img.phaseWeight = eq;
      });
      return;
    }

    this.images.forEach(img => {
      if (type === 'mag') img.magWeight = img.magWeight / total;
      else img.phaseWeight = img.phaseWeight / total;
    });
  }

  /** Returns the normalized mag weight as a percentage string, e.g. "25.00%" */
  magPct(img: ImageSlot): string {
    return (img.magWeight * 100).toFixed(1) + '%';
  }

  phasePct(img: ImageSlot): string {
    return (img.phaseWeight * 100).toFixed(1) + '%';
  }
async onImageSelected(file: File, index: number): Promise<void> {
  this.images[index].file = file;
  this.images[index].viewportData = null;

  try {
    const components = await firstValueFrom(
      this.mixerService.uploadSlot(
        index as 0 | 1 | 2 | 3,
        file,
        this.images[index].magWeight,
        this.images[index].phaseWeight,
      )
    );

    this.images[index].viewportData = {
      originalSrc: components.original,
      ftComponents: {
        magnitude: components.magnitude,
        phase:     components.phase,
        real:      components.real,
        imaginary: components.imaginary,
      },
    };
  } catch (err) {
    console.error(`Failed to upload slot ${index + 1}:`, err);
  }
}

  onComponentChanged(component: FtComponent, index: number): void {
    this.images[index].activeComponent = component;
  }

  onBrightnessContrastChanged(_val: { brightness: number; contrast: number }, _index: number): void { }

  setOutputTarget(index: 0 | 1): void {
    this.outputTarget = index;
    this.outputs.forEach((o, i) => (o.active = i === index));
  }

startMix(): void {
  this.cancel$.next();

  const hasImages = this.images.some(img => img.file !== null);
  if (!hasImages) return;

  this.isMixing = true;
  this.mixProgress = 0;

  const request: MixRequest = {
    componentPair:   this.componentPair,
    regionType:      this.regionType,
    regionSize:      this.regionSize,
    unifyPolicy:     this.unifyPolicy,
    keepAspectRatio: this.keepAspectRatio,
    weights: [
      { magWeight: this.images[0].magWeight, phaseWeight: this.images[0].phaseWeight },
      { magWeight: this.images[1].magWeight, phaseWeight: this.images[1].phaseWeight },
      { magWeight: this.images[2].magWeight, phaseWeight: this.images[2].phaseWeight },
      { magWeight: this.images[3].magWeight, phaseWeight: this.images[3].phaseWeight },
    ],
  };

  this.mixerService
    .runMix(request)
    .pipe(takeUntil(this.cancel$), takeUntil(this.destroy$))
    .subscribe({
      next: (event) => {
        if (event.type === 'progress') {
          this.mixProgress = event.value;

        } else if (event.type === 'result') {
          this.isMixing = false;
          this.mixProgress = 100;

          this.outputs[this.outputTarget].viewportData = {
            originalSrc: event.resultSrc,
            ftComponents: {
              magnitude: event.magnitude,
              phase:     event.phase,
              real:      event.real,
              imaginary: event.imaginary,
            },
          };
        }
      },
      error: (err) => {
        console.error('Mix failed:', err);
        this.isMixing = false;
        this.mixProgress = 0;
      },
    });
}

  cancelMix(): void {
    this.cancel$.next();
    this.isMixing = false;
    this.mixProgress = 0;
  }

  ngOnDestroy(): void {
    this.cancel$.next();
    this.destroy$.next();
    this.destroy$.complete();
  }
}
