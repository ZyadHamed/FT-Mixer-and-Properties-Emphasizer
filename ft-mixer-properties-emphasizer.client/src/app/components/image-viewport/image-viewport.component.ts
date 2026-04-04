import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import {
  Component,
  Input,
  Output,
  EventEmitter,
  ElementRef,
  ViewChild,
  OnChanges,
  SimpleChanges,
  HostListener,
} from '@angular/core';

export type FtComponent = 'image' | 'magnitude' | 'phase' | 'real' | 'imaginary';
export type ViewportMode = 'input' | 'output';

export interface ImageViewportData {
  originalSrc: string | null;
  ftComponents: {
    magnitude: string | null;
    phase: string | null;
    real: string | null;
    imaginary: string | null;
  };
}

@Component({
  imports: [CommonModule, FormsModule],
  standalone: true,
  selector: 'app-image-viewport',
  templateUrl: './image-viewport.component.html',
  styleUrls: ['./image-viewport.component.css'],
})
export class ImageViewportComponent implements OnChanges {
  @Input() label: string = 'IMAGE';
  @Input() mode: ViewportMode = 'input';
  @Input() viewportData: ImageViewportData | null = null;
  @Input() isActive: boolean = false;
  @Input() isAwaiting: boolean = false;
  @Input() regionSize: number = 0; 
  @Input() regionType: 'inner' | 'outer' = 'inner';
  @Output() imageSelected = new EventEmitter<File>();
  @Output() componentChanged = new EventEmitter<FtComponent>();
  @Output() brightnessContrastChanged = new EventEmitter<{ brightness: number; contrast: number }>();

  @ViewChild('fileInput') fileInputRef!: ElementRef<HTMLInputElement>;
  @ViewChild('canvas') canvasRef!: ElementRef<HTMLCanvasElement>;

  selectedComponent: FtComponent = 'image';
  brightness: number = 0;
  contrast: number = 1;

  readonly componentOptions: { value: FtComponent; label: string }[] = [
    { value: 'image', label: 'Image' },
    { value: 'magnitude', label: 'FT Magnitude' },
    { value: 'phase', label: 'FT Phase' },
    { value: 'real', label: 'FT Real' },
    { value: 'imaginary', label: 'FT Imaginary' },
  ];

  
  cssFilter: string = 'brightness(1) contrast(1)';

  private isDragging = false;
  private dragStartX = 0;
  private dragStartY = 0;
  private dragStartBrightness = 0;
  private dragStartContrast = 1;

  private updateCssFilter(): void {
    const b = 1 + this.brightness / 100;
    this.cssFilter = `brightness(${b}) contrast(${this.contrast})`;
  }

  get currentImageSrc(): string | null {
    if (!this.viewportData) return null;

    let data: string | null = null;
    if (this.selectedComponent === 'image') {
      data = this.viewportData.originalSrc;
    } else {
      data = this.viewportData.ftComponents[this.selectedComponent];
    }

    
    if (data && !data.startsWith('data:image')) {
      return `data:image/jpeg;base64,${data}`;
    }
    return data;
  }
  get hasImage(): boolean {
    return !!this.currentImageSrc;
  }
  
  getRegionStyle() {
    const size = this.regionSize;
    
    const offset = (100 - size) / 2;

    return {
      'width': size + '%',
      'height': size + '%',
      'top': offset + '%',
      'left': offset + '%',
      'display': this.selectedComponent === 'image' ? 'none' : 'block'
    };
  }
  ngOnChanges(changes: SimpleChanges): void {
    if (changes['viewportData'] && !this.viewportData) {
      this.selectedComponent = 'image';
      this.resetBrightnessContrast();
    }
  }

  onViewportDoubleClick(): void {
    if (this.mode === 'input') {
      this.fileInputRef.nativeElement.click();
    }
  }

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.imageSelected.emit(input.files[0]);
      input.value = '';
    }
  }

  onComponentChange(value: FtComponent): void {
    this.selectedComponent = value;
    this.componentChanged.emit(value);
  }

  onMouseDown(event: MouseEvent): void {
    if (!this.hasImage) return;
    this.isDragging = true;
    this.dragStartX = event.clientX;
    this.dragStartY = event.clientY;
    this.dragStartBrightness = this.brightness;
    this.dragStartContrast = this.contrast;
    event.preventDefault();
  }

  @HostListener('document:mousemove', ['$event'])
  onMouseMove(event: MouseEvent): void {
    if (!this.isDragging) return;
    const dx = event.clientX - this.dragStartX;
    const dy = event.clientY - this.dragStartY;
    this.brightness = Math.max(-100, Math.min(100, this.dragStartBrightness - dy * 0.5));
    this.contrast = Math.max(0.1, Math.min(3, this.dragStartContrast + dx * 0.01));
    this.updateCssFilter(); 
    this.brightnessContrastChanged.emit({ brightness: this.brightness, contrast: this.contrast });
  }

  @HostListener('document:mouseup')
  onMouseUp(): void {
    this.isDragging = false;
  }

  resetBrightnessContrast(): void {
    this.brightness = 0;
    this.contrast = 1;
    this.updateCssFilter(); 
    this.brightnessContrastChanged.emit({ brightness: 0, contrast: 1 });
  }
}
