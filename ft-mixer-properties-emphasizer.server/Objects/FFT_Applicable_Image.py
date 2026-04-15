import io
import base64
import numpy as np
from PIL import Image
from typing import Optional


class FFT_Applicable_Image:
    """
    Core image container. Accepts raw bytes, a base64 string, or a numpy array.
    Owns the source array and can produce all FFT display components on demand.
    """

    # ──────────────────────────────────────────────────────────────────────
    # Construction
    # ──────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        image_bytes: Optional[bytes] = None,
        base64_str:  Optional[str]   = None,
        array:       Optional[np.ndarray] = None,
    ):
        """
        Priority: array > image_bytes > base64_str.
        The stored array is always float64 (or complex128), range 0–255 for
        spatial images so that FFT scale stays consistent with the rest of
        the pipeline.
        """
        if array is not None:
            self._array = self._coerce(array)
        elif image_bytes is not None:
            self._array = self._from_bytes(image_bytes)
        elif base64_str is not None:
            self._array = self._from_bytes(base64.b64decode(base64_str))
        else:
            raise ValueError(
                "Provide at least one of: image_bytes, base64_str, array."
            )

        # Lazy cache so we only run fft2 once per array version
        self._fft_cache: Optional[np.ndarray] = None

    # ──────────────────────────────────────────────────────────────────────
    # Public array access
    # ──────────────────────────────────────────────────────────────────────

    @property
    def array(self) -> np.ndarray:
        return self._array

    @array.setter
    def array(self, new_arr: np.ndarray):
        self._array = self._coerce(new_arr)
        self._fft_cache = None          # invalidate cache

    @property
    def shape(self):
        return self._array.shape

    def is_complex(self) -> bool:
        return np.iscomplexobj(self._array)

    # ──────────────────────────────────────────────────────────────────────
    # FFT helpers
    # ──────────────────────────────────────────────────────────────────────

    def get_raw_fft(self) -> np.ndarray:
        """Returns (and caches) the raw 2-D FFT of the stored array."""
        if self._fft_cache is None:
            self._fft_cache = np.fft.fft2(self._array)
        return self._fft_cache

    def get_fft_components(self) -> dict:
        """
        Returns a dict with all 8 display-ready FFT images (0.0–1.0 float64)
        plus the raw FFT array.

        Keys
        ----
        raw_fft,
        shifted_mag, shifted_phase, shifted_real, shifted_imag,
        unshifted_mag, unshifted_phase, unshifted_real, unshifted_imag
        """
        F = self.get_raw_fft()
        (
            raw_fft,
            shifted_mag,   shifted_phase,   shifted_real,   shifted_imag,
            unshifted_mag, unshifted_phase, unshifted_real, unshifted_imag,
        ) = self._prepare_fft_for_display(F)

        return {
            "raw_fft":          raw_fft,
            "shifted_mag":      shifted_mag,
            "shifted_phase":    shifted_phase,
            "shifted_real":     shifted_real,
            "shifted_imag":     shifted_imag,
            "unshifted_mag":    unshifted_mag,
            "unshifted_phase":  unshifted_phase,
            "unshifted_real":   unshifted_real,
            "unshifted_imag":   unshifted_imag,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Serialisation helpers
    # ──────────────────────────────────────────────────────────────────────

    def to_bytes(self, fmt: str = "JPEG") -> bytes:
        display = np.abs(self._array) if self.is_complex() else self._array
        d_min, d_max = display.min(), display.max()
        if d_max > d_min:
            display = (display - d_min) / (d_max - d_min)
        uint8 = np.clip(display * 255, 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(uint8).save(buf, format=fmt, quality=95)
        return buf.getvalue()

    def to_base64(self, fmt: str = "JPEG") -> str:
        return base64.b64encode(self.to_bytes(fmt)).decode("utf-8")

    # ──────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _from_bytes(data: bytes) -> np.ndarray:
        pil = Image.open(io.BytesIO(data)).convert("L")
        # Store in 0–255 range (float64) to keep FFT scale consistent
        return np.array(pil, dtype=np.float64)

    @staticmethod
    def _coerce(arr: np.ndarray) -> np.ndarray:
        """Ensure the array is float64 or complex128."""
        if np.iscomplexobj(arr):
            return arr.astype(np.complex128)
        return arr.astype(np.float64)

    # ── FFT display pipeline (moved from FrequencyTransformService) ────────

    @staticmethod
    def _prepare_fft_for_display(F: np.ndarray):
        """
        Full FFT display pipeline: produces shifted + unshifted versions of
        magnitude, phase, real and imaginary, all normalised to 0.0–1.0.
        """
        scale = 100_000.0

        def _process(F_arr):
            mag   = np.abs(F_arr)
            phase = np.angle(F_arr)
            real  = np.real(F_arr)
            imag  = np.imag(F_arr)

            # Magnitude – log scale
            mag_log = np.log(1 + mag * scale)
            mag_max = mag_log.max()
            disp_mag = mag_log / mag_max if mag_max > 0 else mag_log

            # Phase – linear [-π, π] → [0, 1]
            disp_phase = (phase + np.pi) / (2 * np.pi)

            # Real – signed log + min-max
            real_log = np.sign(real) * np.log(1 + np.abs(real * scale))
            r_min, r_max = real_log.min(), real_log.max()
            disp_real = (
                (real_log - r_min) / (r_max - r_min)
                if r_max > r_min else real_log
            )

            # Imaginary – signed log + min-max
            imag_log = np.sign(imag) * np.log(1 + np.abs(imag * scale))
            i_min, i_max = imag_log.min(), imag_log.max()
            disp_imag = (
                (imag_log - i_min) / (i_max - i_min)
                if i_max > i_min else imag_log
            )

            return disp_mag, disp_phase, disp_real, disp_imag

        F_shifted = np.fft.fftshift(F)
        s_mag,  s_phase,  s_real,  s_imag  = _process(F_shifted)
        u_mag,  u_phase,  u_real,  u_imag  = _process(F)

        return (
            F,
            s_mag, s_phase, s_real, s_imag,
            u_mag, u_phase, u_real, u_imag,
        )