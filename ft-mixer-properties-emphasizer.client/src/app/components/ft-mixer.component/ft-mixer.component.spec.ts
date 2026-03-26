import { ComponentFixture, TestBed } from '@angular/core/testing';

import { FtMixerComponent } from './ft-mixer.component';

describe('FtMixerComponent', () => {
  let component: FtMixerComponent;
  let fixture: ComponentFixture<FtMixerComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [FtMixerComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(FtMixerComponent);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
