import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ModeHostComponent } from './mode-host.component';

describe('ModeHostComponent', () => {
  let component: ModeHostComponent;
  let fixture: ComponentFixture<ModeHostComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ModeHostComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ModeHostComponent);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
