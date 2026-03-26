import { Component } from '@angular/core';
import { trigger, transition, style, animate } from '@angular/animations';
import { FtEmphasizerComponent } from '../ft-emphasizer.component/ft-emphasizer.component';
import { FtMixerComponent } from '../ft-mixer.component/ft-mixer.component';

@Component({
  selector: 'app-mode-host',
  templateUrl: './mode-host.component.html',
  styleUrls: ['./mode-host.component.css'],
  animations: [
    trigger('fadeSlide', [
      transition(':enter', [
        style({ opacity: 0, transform: 'translateY(8px)' }),
        animate('220ms ease-out', style({ opacity: 1, transform: 'translateY(0)' }))
      ]),
      transition(':leave', [
        animate('150ms ease-in', style({ opacity: 0, transform: 'translateY(-8px)' }))
      ])
    ])
  ],
  imports: [FtEmphasizerComponent, FtMixerComponent]
})
export class ModeHostComponent {
  activeMode: 'mixer' | 'emphasizer' = 'mixer';

  setMode(mode: 'mixer' | 'emphasizer'): void {
    this.activeMode = mode;
  }
}