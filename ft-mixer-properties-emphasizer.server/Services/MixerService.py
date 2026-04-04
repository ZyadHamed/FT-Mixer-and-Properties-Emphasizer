import base64
import io
import numpy as np
from PIL import Image

from Services.ImageProcessingService import load_and_unify_images


class MixerService:
    """
    Weighted FT mixing of up to 4 images.

    Supports:
      - Component pairs : 'mag-phase'  |  'real-imag'
      - Region selection: 'inner' (low-freq)  |  'outer' (high-freq)
    """

    def mix(
        self,
        images: list[dict],
        component_pair: str,
        region_type: str,
        region_size: float,
        unify_policy: str,
        keep_aspect_ratio: bool,
    ) -> tuple[str, np.ndarray]:
        """
        Returns:
          - result_b64   : base64-encoded JPEG of the spatial result
          - spatial_arr  : uint8 numpy array of the result (needed by endpoint
                           to compute FT components without re-decoding JPEG)
        """
        arrays = load_and_unify_images(
            images_bytes=[img["bytes"] for img in images],
            unify_policy=unify_policy,
            keep_aspect_ratio=keep_aspect_ratio,
        )

        h, w = arrays[0].shape
        mask = self._build_mask(h, w, region_size, region_type)

        ffts_shifted = [np.fft.fftshift(np.fft.fft2(arr)) for arr in arrays]

        if component_pair == "mag-phase":
            result_fft = self._mix_mag_phase(ffts_shifted, images, mask)
        else:
            result_fft = self._mix_real_imag(ffts_shifted, images, mask)

        spatial = np.abs(np.fft.ifft2(np.fft.ifftshift(result_fft)))
        spatial_arr = np.clip(spatial, 0, 255).astype(np.uint8)

        result_b64 = self._array_to_b64(spatial_arr)
        return result_b64, spatial_arr

    # ── Mixing strategies ──────────────────────────────────────────────────
    
    def _mix_mag_phase(self, ffts_shifted, images, mask):
        h, w = ffts_shifted[0].shape
        mixed_mag   = np.zeros((h, w), dtype=np.float64)
        mixed_phase = np.zeros((h, w), dtype=np.float64)

        for fft_s, img in zip(ffts_shifted, images):
            mixed_mag   += img["mag_weight"]   * np.abs(fft_s)
            mixed_phase += img["phase_weight"] * np.angle(fft_s)

        combined = mixed_mag * np.exp(1j * mixed_phase)
        
        return combined * mask

    def _mix_real_imag(self, ffts_shifted, images, mask):
        h, w = ffts_shifted[0].shape
        mixed_real = np.zeros((h, w), dtype=np.float64)
        mixed_imag = np.zeros((h, w), dtype=np.float64)

        for fft_s, img in zip(ffts_shifted, images):
            mixed_real += img["mag_weight"]   * np.real(fft_s)
            mixed_imag += img["phase_weight"] * np.imag(fft_s)

        return (mixed_real + 1j * mixed_imag) * mask

    # ── Region mask ────────────────────────────────────────────────────────
    @staticmethod
    def _build_mask(h, w, region_size, region_type):
        
        mask = np.zeros((h, w), dtype=np.float64)
        cy, cx = h // 2, w // 2
        
       
        rh = int((h * region_size / 100) / 2)
        rw = int((w * region_size / 100) / 2)
        
        
        mask[cy - rh : cy + rh, cx - rw : cx + rw] = 1.0
        
        if region_type == "outer":
            mask = 1.0 - mask
            
        return mask
    @staticmethod
    def _array_to_b64(arr: np.ndarray) -> str:
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=95)
        return base64.b64encode(buf.getvalue()).decode("utf-8")