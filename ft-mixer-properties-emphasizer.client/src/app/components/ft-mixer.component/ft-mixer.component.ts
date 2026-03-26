// ft-mixer.component.ts
import { Component, OnDestroy } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DecimalPipe, UpperCasePipe} from '@angular/common';

export interface ImageSlot {
  loaded: boolean;
  displaySrc: string | null;
  displayMode: 'image' | 'magnitude' | 'phase' | 'real' | 'imaginary';
  magWeight: number;
  phaseWeight: number;
  weight: number;
  brightness: number;
  contrast: number;
  color: string;
}

export interface OutputSlot {
  hasResult: boolean;
  resultSrc: string | null;
  active: boolean;
}

@Component({
  selector: 'app-ft-mixer',
  templateUrl: './ft-mixer.component.html',
  styleUrls: ['./ft-mixer.component.css'],
  imports: [FormsModule, DecimalPipe, UpperCasePipe],
})
export class FtMixerComponent implements OnDestroy {

  images: ImageSlot[] = [
    { loaded: false, displaySrc: null, displayMode: 'image', magWeight: 1, phaseWeight: 1, weight: 1, brightness: 1, contrast: 1, color: '#4a9eff' },
    { loaded: false, displaySrc: null, displayMode: 'image', magWeight: 1, phaseWeight: 1, weight: 1, brightness: 1, contrast: 1, color: '#3ecf8e' },
    { loaded: false, displaySrc: null, displayMode: 'image', magWeight: 1, phaseWeight: 1, weight: 1, brightness: 1, contrast: 1, color: '#f7c948' },
    { loaded: false, displaySrc: null, displayMode: 'image', magWeight: 1, phaseWeight: 1, weight: 1, brightness: 1, contrast: 1, color: '#9b8dff' },
  ];

  outputs: OutputSlot[] = [
    { hasResult: false, resultSrc: null, active: true },
    { hasResult: false, resultSrc: null, active: false },
  ];

  regionSize: number = 40;
  isMixing: boolean = false;
  mixProgress: number = 0;

  private mixCancelFlag = false;
  private mixInterval: any = null;

  browseImage(index: number): void {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.onchange = (e: Event) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (ev) => {
          this.images[index].displaySrc = ev.target?.result as string;
          this.images[index].loaded = true;
        };
        reader.readAsDataURL(file);
      }
    };
    input.click();
  }

  startBrightnessAdjust(event: MouseEvent, index: number): void {
    const startX = event.clientX;
    const startY = event.clientY;
    const startBrightness = this.images[index].brightness;
    const startContrast = this.images[index].contrast;

    const onMove = (e: MouseEvent) => {
      const dx = (e.clientX - startX) / 200;
      const dy = (e.clientY - startY) / 200;
      this.images[index].brightness = Math.max(0.1, Math.min(3, startBrightness - dy));
      this.images[index].contrast   = Math.max(0.1, Math.min(3, startContrast   + dx));
    };

    const onUp = () => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    };

    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
    event.preventDefault();
  }

  startMix(): void {
    if (this.isMixing) {
      this.cancelMix();
      setTimeout(() => this.runMix(), 100);
    } else {
      this.runMix();
    }
  }

  private runMix(): void {
    this.isMixing = true;
    this.mixProgress = 0;
    this.mixCancelFlag = false;

    // Simulated progress — replace with actual backend call
    this.mixInterval = setInterval(() => {
      if (this.mixCancelFlag) {
        clearInterval(this.mixInterval);
        this.isMixing = false;
        this.mixProgress = 0;
        return;
      }
      this.mixProgress += 5;
      if (this.mixProgress >= 100) {
        clearInterval(this.mixInterval);
        this.isMixing = false;
        this.mixProgress = 100;
        // TODO: set output result from backend response
      }
    }, 100);
  }

  cancelMix(): void {
    this.mixCancelFlag = true;
    clearInterval(this.mixInterval);
    this.isMixing = false;
    this.mixProgress = 0;
  }

  ngOnDestroy(): void {
    this.cancelMix();
  }
}