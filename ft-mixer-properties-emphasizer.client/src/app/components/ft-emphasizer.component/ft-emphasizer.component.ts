// ft-emphasizer.component.ts
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DecimalPipe, UpperCasePipe } from '@angular/common';

export type EmphasizerAction =
  | 'shift' | 'stretch' | 'mirror' | 'rotate'
  | 'even'  | 'odd'
  | 'complex_exp' | 'window'
  | 'differentiate' | 'integrate'
  | 'ft_repeat';

export interface EmphasizerParams {
  // Shift
  shiftX: number;
  shiftY: number;
  // Stretch
  scaleX: number;
  scaleY: number;
  // Rotate
  angle: number;
  expandCanvas: boolean;
  // Mirror
  mirrorAxis: 'horizontal' | 'vertical' | 'both';
  // Complex exponential
  freqU: number;
  freqV: number;
  // Window
  windowType: 'rectangular' | 'gaussian' | 'hamming' | 'hanning';
  sigma: number;
  windowW: number;
  windowH: number;
  // Repeated FT
  ftRepeat: number;
  // Chained FT on top
  chainFT: number;
}

@Component({
  selector: 'app-ft-emphasizer',
  templateUrl: './ft-emphasizer.component.html',
  styleUrls: ['./ft-emphasizer.component.css'],
  imports: [FormsModule, DecimalPipe, UpperCasePipe],
})
export class FtEmphasizerComponent {

  selectedAction: EmphasizerAction = 'shift';
  domain: 'spatial' | 'frequency' = 'spatial';

  params: EmphasizerParams = {
    shiftX: 0, shiftY: 0,
    scaleX: 1, scaleY: 1,
    angle: 0, expandCanvas: true,
    mirrorAxis: 'horizontal',
    freqU: 0, freqV: 0,
    windowType: 'gaussian', sigma: 30, windowW: 256, windowH: 256,
    ftRepeat: 1,
    chainFT: 0,
  };

  // Image state
  originalLoaded = false;
  resultReady    = false;
  originalSrc:  string | null = null;
  resultSrc:    string | null = null;
  ftOrigSrc:    string | null = null;
  ftResultSrc:  string | null = null;

  // Display modes per viewport
  spatialOrigMode  = 'image';
  spatialResultMode = 'image';
  ftOrigMode       = 'magnitude';
  ftResultMode     = 'magnitude';

  onActionChange(): void {
    // Optionally reset params or update UI reactively here
  }

  browseImage(): void {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.onchange = (e: Event) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (ev) => {
          this.originalSrc = ev.target?.result as string;
          this.originalLoaded = true;
          // TODO: trigger backend to compute FT of original
          // this.ftOrigSrc = ...
        };
        reader.readAsDataURL(file);
      }
    };
    input.click();
  }

  applyAction(): void {
    if (!this.originalLoaded) return;

    const payload = {
      action: this.selectedAction,
      domain: this.domain,
      params: this.params,
      image: this.originalSrc,
    };

    // TODO: call backend service
    // this.ftService.applyAction(payload).subscribe(result => {
    //   this.resultSrc    = result.spatialImage;
    //   this.ftResultSrc  = result.ftImage;
    //   this.resultReady  = true;
    // });

    console.log('Applying action:', payload);
    // Placeholder until backend is wired
    this.resultReady = false;
  }
}