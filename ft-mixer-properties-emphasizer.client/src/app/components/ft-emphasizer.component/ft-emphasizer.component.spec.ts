import { ComponentFixture, TestBed } from '@angular/core/testing';

import { FtEmphasizerComponent } from './ft-emphasizer.component';

describe('FtEmphasizerComponent', () => {
  let component: FtEmphasizerComponent;
  let fixture: ComponentFixture<FtEmphasizerComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [FtEmphasizerComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(FtEmphasizerComponent);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
