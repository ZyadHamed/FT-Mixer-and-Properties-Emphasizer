import io
import base64
import numpy as np
from PIL import Image
from typing import Optional

from .FFT_Applicable_Image import FFT_Applicable_Image


class FourImagesMixer:
    """
    Holds up to 4 FFT_Applicable_Image slots.
    Each slot is updated independently when a new image is uploaded,
    so the FFT is never recomputed unless the source image changes.

    Component pairs
    ---------------
    'mag-phase'  – mix magnitude from one set of weights, phase from another
    'real-imag'  – mix real part and imaginary part independently

    Region types
    ------------
    'inner' – low-frequency (DC-centred) rectangle kept
    'outer' – high-frequency ring kept  (complement of inner)
    """

    NUM_SLOTS = 4

    def __init__(self):
        self._slots: list[Optional[_WeightedSlot]] = [None] * self.NUM_SLOTS

    # ──────────────────────────────────────────────────────────────────────
    # Slot management
    # ──────────────────────────────────────────────────────────────────────

    def set_image(
        self,
        index:        int,
        image_bytes:  Optional[bytes] = None,
        base64_str:   Optional[str]   = None,
        mag_weight:   float = 1.0,
        phase_weight: float = 1.0,
    ) -> None:
        """
        Load / replace one of the four image slots.
        The FFT is computed immediately and cached inside FFT_Applicable_Image.
        """
        if not 0 <= index < self.NUM_SLOTS:
            raise IndexError(f"Slot index must be 0–{self.NUM_SLOTS - 1}.")
        img = FFT_Applicable_Image(
            image_bytes=image_bytes,
            base64_str=base64_str,
        )
        self._slots[index] = _WeightedSlot(
            image=img,
            mag_weight=mag_weight,
            phase_weight=phase_weight,
        )

    def update_weights(
        self,
        index:        int,
        mag_weight:   float,
        phase_weight: float,
    ) -> None:
        """Change mixing weights without reloading the image."""
        slot = self._slots[index]
        if slot is None:
            raise ValueError(f"Slot {index} is empty.")
        slot.mag_weight   = mag_weight
        slot.phase_weight = phase_weight

    def clear_slot(self, index: int) -> None:
        self._slots[index] = None

    # ──────────────────────────────────────────────────────────────────────
    # Core mix
    # ──────────────────────────────────────────────────────────────────────

    def mix(
        self,
        component_pair:    str   = "mag-phase",
        region_type:       str   = "inner",
        region_size:       float = 40.0,
        unify_policy:      str   = "smallest",
        keep_aspect_ratio: bool  = True,
    ) -> dict:
        """
        Run the weighted FT mix across all populated slots.

        Returns a dict with:
          result_b64   – base64 JPEG of the spatial result
          spatial_arr  – uint8 numpy array of the spatial result
          magnitude / phase / real / imaginary – base64 FT display images
        """
        active = [s for s in self._slots if s is not None]
        if not active:
            raise ValueError("No images loaded.")

        arrays = self._unify_arrays(
            [s.image.array for s in active],
            unify_policy,
            keep_aspect_ratio,
        )

        h, w  = arrays[0].shape
        mask  = self._build_mask(h, w, region_size, region_type)

        # Recompute FFTs on the *unified* arrays (sizes may differ from
        # the originally loaded images, so we build temporary wrappers)
        from .FFT_Applicable_Image import FFT_Applicable_Image  # local to avoid circulars
        ffts_shifted = [
            np.fft.fftshift(np.fft.fft2(arr)) for arr in arrays
        ]

        if component_pair == "mag-phase":
            result_fft = self._mix_mag_phase(ffts_shifted, active, mask)
        else:
            result_fft = self._mix_real_imag(ffts_shifted, active, mask)

        spatial = np.abs(np.fft.ifft2(np.fft.ifftshift(result_fft)))
        spatial_arr = np.clip(spatial, 0, 255).astype(np.uint8)

        # FT display of the result
        tmp = FFT_Applicable_Image(array=spatial_arr.astype(np.float64))
        fft_components = tmp.get_fft_components()

        def _to_b64(a: np.ndarray) -> str:
            uint8 = np.clip(a * 255, 0, 255).astype(np.uint8)
            buf   = io.BytesIO()
            Image.fromarray(uint8).save(buf, format="JPEG", quality=95)
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        result_b64 = self._array_to_b64(spatial_arr)

        return {
            "result_image": result_b64,
            "spatial_arr":  spatial_arr,
            "magnitude":    _to_b64(fft_components["shifted_mag"]),
            "phase":        _to_b64(fft_components["shifted_phase"]),
            "real":         _to_b64(fft_components["shifted_real"]),
            "imaginary":    _to_b64(fft_components["shifted_imag"]),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Mixing strategies
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _mix_mag_phase(
        ffts_shifted: list[np.ndarray],
        slots:        list["_WeightedSlot"],
        mask:         np.ndarray,
    ) -> np.ndarray:
        h, w = ffts_shifted[0].shape
        mixed_mag   = np.zeros((h, w), dtype=np.float64)
        mixed_phase = np.zeros((h, w), dtype=np.float64)
        for fft_s, slot in zip(ffts_shifted, slots):
            mixed_mag   += slot.mag_weight   * np.abs(fft_s)
            mixed_phase += slot.phase_weight * np.angle(fft_s)
        return mixed_mag * np.exp(1j * mixed_phase) * mask

    @staticmethod
    def _mix_real_imag(
        ffts_shifted: list[np.ndarray],
        slots:        list["_WeightedSlot"],
        mask:         np.ndarray,
    ) -> np.ndarray:
        h, w = ffts_shifted[0].shape
        mixed_real = np.zeros((h, w), dtype=np.float64)
        mixed_imag = np.zeros((h, w), dtype=np.float64)
        for fft_s, slot in zip(ffts_shifted, slots):
            mixed_real += slot.mag_weight   * np.real(fft_s)
            mixed_imag += slot.phase_weight * np.imag(fft_s)
        return (mixed_real + 1j * mixed_imag) * mask

    # ──────────────────────────────────────────────────────────────────────
    # Region mask
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_mask(
        h:           int,
        w:           int,
        region_size: float,
        region_type: str,
    ) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.float64)
        cy, cx = h // 2, w // 2
        rh = int((h * region_size / 100) / 2)
        rw = int((w * region_size / 100) / 2)
        mask[cy - rh: cy + rh, cx - rw: cx + rw] = 1.0
        if region_type == "outer":
            mask = 1.0 - mask
        return mask

    # ──────────────────────────────────────────────────────────────────────
    # Image unification (replaces load_and_unify_images for in-memory arrays)
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _unify_arrays(
        arrays:            list[np.ndarray],
        policy:            str,
        keep_aspect_ratio: bool,
    ) -> list[np.ndarray]:
        """
        Resize all arrays to a unified (H, W) according to policy.
        Works directly on numpy arrays—no round-trip through bytes.
        """
        shapes = [(a.shape[1], a.shape[0]) for a in arrays]  # (W, H)

        if policy == "smallest":
            target_wh = min(shapes, key=lambda s: s[0] * s[1])
        elif policy == "largest":
            target_wh = max(shapes, key=lambda s: s[0] * s[1])
        else:
            target_wh = shapes[0]

        target_w, target_h = target_wh
        result = []

        for arr in arrays:
            pil = Image.fromarray(
                np.clip(arr, 0, 255).astype(np.uint8)
            ).convert("L")

            if keep_aspect_ratio:
                pil.thumbnail((target_w, target_h), Image.LANCZOS)
                canvas = Image.new("L", (target_w, target_h), 0)
                offset = (
                    (target_w - pil.width)  // 2,
                    (target_h - pil.height) // 2,
                )
                canvas.paste(pil, offset)
                result.append(np.array(canvas, dtype=np.float64))
            else:
                pil_r = pil.resize((target_w, target_h), Image.LANCZOS)
                result.append(np.array(pil_r, dtype=np.float64))

        return result

    # ──────────────────────────────────────────────────────────────────────
    # Misc
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _array_to_b64(arr: np.ndarray) -> str:
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=95)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Internal data-holder (not exported)
# ──────────────────────────────────────────────────────────────────────────────

class _WeightedSlot:
    __slots__ = ("image", "mag_weight", "phase_weight")

    def __init__(
        self,
        image:        FFT_Applicable_Image,
        mag_weight:   float,
        phase_weight: float,
    ):
        self.image        = image
        self.mag_weight   = mag_weight
        self.phase_weight = phase_weight