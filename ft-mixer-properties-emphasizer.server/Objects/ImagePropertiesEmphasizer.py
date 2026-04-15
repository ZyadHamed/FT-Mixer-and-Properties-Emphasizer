import numpy as np
from scipy.ndimage import zoom, rotate as ndimage_rotate, convolve

from .FFT_Applicable_Image import FFT_Applicable_Image


class ImagePropertiesEmphasizer(FFT_Applicable_Image):
    """
    Extends FFT_Applicable_Image with every spatial / frequency operation
    that is exposed by the endpoints.

    Each method returns a *new* ImagePropertiesEmphasizer containing the
    result so that operations can be chained and the original is never
    mutated.  The caller can also call .array on the result directly.
    """

    # ──────────────────────────────────────────────────────────────────────
    # Factory helper
    # ──────────────────────────────────────────────────────────────────────

    def _wrap(self, arr: np.ndarray) -> "ImagePropertiesEmphasizer":
        """Wrap a result array in a fresh instance (no I/O overhead)."""
        return ImagePropertiesEmphasizer(array=arr)

    # ──────────────────────────────────────────────────────────────────────
    # Spatial operations  (mirror the standalone functions 1-to-1)
    # ──────────────────────────────────────────────────────────────────────

    def shift(
        self,
        shift_x: int = 0,
        shift_y: int = 0,
        cyclic:  bool = False,
        flip:    bool = True,
    ) -> "ImagePropertiesEmphasizer":
        arr = self._array
        if flip:
            shift_y = -shift_y

        shifted = np.roll(arr, shift=shift_y, axis=0)
        shifted = np.roll(shifted, shift=shift_x, axis=1)

        if not cyclic:
            if shift_y > 0:
                shifted[:shift_y, :] = 0
            elif shift_y < 0:
                shifted[shift_y:, :] = 0
            if shift_x > 0:
                shifted[:, :shift_x] = 0
            elif shift_x < 0:
                shifted[:, shift_x:] = 0

        return self._wrap(shifted)

    def multiply_by_complex_exponential(
        self,
        amplitude: float,
        freq_u: float,
        freq_v: float,
    ) -> "ImagePropertiesEmphasizer":
        arr = self._array
        H, W = arr.shape
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        kernel = amplitude * np.exp(1j * 2 * np.pi * (freq_u * xx + freq_v * yy))
        return self._wrap(arr * kernel)

    def stretch(
        self,
        stretch_x: float,
        stretch_y: float,
        order: int = 3,
    ) -> "ImagePropertiesEmphasizer":
        arr = self._array
        factors = (stretch_y, stretch_x)
        if np.iscomplexobj(arr):
            real_s = zoom(arr.real, factors, order=order)
            imag_s = zoom(arr.imag, factors, order=order)
            return self._wrap(real_s + 1j * imag_s)
        return self._wrap(zoom(arr, factors, order=order))

    def mirror(
        self,
        mirror_axis:    str  = "horizontal",
        duplicate_mode: bool = True,
    ) -> "ImagePropertiesEmphasizer":
        arr = self._array
        if not duplicate_mode:
            if mirror_axis == "horizontal":
                return self._wrap(np.flipud(arr))
            elif mirror_axis == "vertical":
                return self._wrap(np.fliplr(arr))
            return self._wrap(np.flipud(np.fliplr(arr)))
        else:
            if mirror_axis == "horizontal":
                return self._wrap(np.vstack([arr, np.flipud(arr)]))
            elif mirror_axis == "vertical":
                return self._wrap(np.hstack([arr, np.fliplr(arr)]))
            top    = np.hstack([arr, np.fliplr(arr)])
            bottom = np.hstack([np.flipud(arr), np.flipud(np.fliplr(arr))])
            return self._wrap(np.vstack([top, bottom]))

    def make_even_or_odd(
        self,
        symmetry_type: str,
    ) -> "ImagePropertiesEmphasizer":
        arr = self._coerce(self._array)
        if np.iscomplexobj(arr):
            img_centered, mean = arr, 0
        else:
            mean = np.mean(arr)
            img_centered = arr - mean

        f_neg = np.roll(
            np.flip(img_centered, axis=(0, 1)),
            shift=(1, 1),
            axis=(0, 1),
        )
        if symmetry_type == "even":
            return self._wrap(0.5 * (img_centered + f_neg) + mean)
        elif symmetry_type == "odd":
            return self._wrap(0.5 * (img_centered - f_neg))
        raise ValueError("symmetry_type must be 'even' or 'odd'")

    def rotate(
        self,
        angle: float,
        order: int = 3,
    ) -> "ImagePropertiesEmphasizer":
        arr = self._array
        if np.iscomplexobj(arr):
            real_r = ndimage_rotate(arr.real, angle, reshape=True, order=order)
            imag_r = ndimage_rotate(arr.imag, angle, reshape=True, order=order)
            return self._wrap(real_r + 1j * imag_r)
        return self._wrap(ndimage_rotate(arr, angle, reshape=True, order=order))

    def differentiate(
        self,
        axis:   str = "x",
        method: str = "central",
    ) -> "ImagePropertiesEmphasizer":
        arr = self._coerce(self._array)
        if method == "central":
            kernel = (
                np.array([[1, 0, -1]]) / 2.0
                if axis == "x"
                else np.array([[1], [0], [-1]]) / 2.0
            )
        elif method == "sobel":
            kernel = (
                np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0
                if axis == "x"
                else np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8.0
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        if np.iscomplexobj(arr):
            real_d = convolve(arr.real, kernel, mode="reflect")
            imag_d = convolve(arr.imag, kernel, mode="reflect")
            return self._wrap(real_d + 1j * imag_d)

        raw = convolve(arr.real, kernel, mode="reflect")
        d_min, d_max = raw.min(), raw.max()
        if d_max > d_min:
            normed = ((raw - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            normed = raw.astype(np.uint8)
        return self._wrap(normed.astype(np.float64))

    def integrate(self, axis: str = "x") -> "ImagePropertiesEmphasizer":
        if axis == "x":
            return self._wrap(np.cumsum(self._array, axis=1))
        elif axis == "y":
            return self._wrap(np.cumsum(self._array, axis=0))
        raise ValueError("axis must be 'x' or 'y'")

    def multiply_by_window(
        self,
        window_width:  int,
        window_height: int,
        center_x:      int,
        center_y:      int,
        window_type:   str = "hamming",
        **kwargs,
    ) -> "ImagePropertiesEmphasizer":
        arr   = self._array
        rows, cols = arr.shape

        start_y = center_y - window_height // 2
        end_y   = start_y + window_height
        start_x = center_x - window_width  // 2
        end_x   = start_x + window_width

        if start_y < 0 or end_y > rows or start_x < 0 or end_x > cols:
            raise ValueError(
                f"Window ({window_width}×{window_height}) centred at "
                f"({center_x},{center_y}) exceeds image ({cols}×{rows})."
            )

        if window_type == "rectangular":
            win_y, win_x = np.ones(window_height), np.ones(window_width)
        elif window_type == "hamming":
            win_y, win_x = np.hamming(window_height), np.hamming(window_width)
        elif window_type == "hanning":
            win_y, win_x = np.hanning(window_height), np.hanning(window_width)
        elif window_type == "gaussian":
            sig_y = kwargs.get("sigma_y", window_height / 6)
            sig_x = kwargs.get("sigma_x", window_width  / 6)
            y = np.arange(window_height) - window_height / 2
            x = np.arange(window_width)  - window_width  / 2
            win_y = np.exp(-(y ** 2) / (2 * sig_y ** 2))
            win_x = np.exp(-(x ** 2) / (2 * sig_x ** 2))
        else:
            raise ValueError(f"Unsupported window type: {window_type}")

        window_2d = np.outer(win_y, win_x)
        padded    = np.zeros((rows, cols), dtype=np.float64)
        padded[start_y:end_y, start_x:end_x] = window_2d

        return self._wrap(arr * padded)

    # ──────────────────────────────────────────────────────────────────────
    # Complex-spatial display helper  (used by complex_exp endpoint)
    # ──────────────────────────────────────────────────────────────────────

    def get_complex_spatial_components(self) -> dict:
        """
        For a complex spatial array (e.g. after multiply_by_complex_exponential)
        return the 4 display-ready components normalised to 0.0–1.0.
        """
        arr = self._array
        mag   = np.abs(arr)
        phase = np.angle(arr)
        real  = np.real(arr)
        imag  = np.imag(arr)

        def _minmax(a):
            lo, hi = a.min(), a.max()
            return (a - lo) / (hi - lo) if hi > lo else a

        return {
            "magnitude":  _minmax(mag),
            "phase":      (phase + np.pi) / (2 * np.pi),
            "real":       _minmax(real),
            "imaginary":  _minmax(imag),
        }