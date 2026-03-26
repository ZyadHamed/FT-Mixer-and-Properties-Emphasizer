import { Component, signal } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { ModeHostComponent } from './components/mode-host.component/mode-host.component';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, ModeHostComponent],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App {
  protected readonly title = signal('ft-mixer-properties-emphasizer');
}
