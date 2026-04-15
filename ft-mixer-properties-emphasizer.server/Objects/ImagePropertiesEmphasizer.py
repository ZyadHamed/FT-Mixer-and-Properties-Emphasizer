import numpy as np
from scipy.ndimage import zoom, rotate as ndimage_rotate, convolve

from .FFT_Applicable_Image import FFT_Applicable_Image


class ImagePropertiesEmphasizer(FFT_Applicable_Image):

    # ──────────────────────────────────────────────────────────────────────
    # Factory helper
    # ──────────────────────────────────────────────────────────────────────

    def _wrap(self, arr: np.ndarray) -> "ImagePropertiesEmphasizer":
        return ImagePropertiesEmphasizer(array=arr)

    # ──────────────────────────────────────────────────────────────────────
    # Frequency-domain execution helper
    # ──────────────────────────────────────────────────────────────────────

    def _apply_in_frequency_domain(
        self,
        operation,          # callable: (ImagePropertiesEmphasizer) -> ImagePropertiesEmphasizer
    ) -> "ImagePropertiesEmphasizer":
        """
        Shared wrapper for in_frequency_domain=True on every operation:
          1. FFT the stored array and center DC
          2. Apply the operation on the centered FFT
          3. ifftshift → ifft2 → return real spatial result
        """
        fft_centered = np.fft.fftshift(self.get_raw_fft())
        fft_img      = ImagePropertiesEmphasizer(array=fft_centered)
        result_fft   = operation(fft_img)
        spatial      = np.fft.ifft2(np.fft.ifftshift(result_fft.array))
        return self._wrap(spatial)

    # ──────────────────────────────────────────────────────────────────────
    # Operations
    # ──────────────────────────────────────────────────────────────────────

    def shift(
        self,
        shift_x: int  = 0,
        shift_y: int  = 0,
        cyclic:  bool = False,
        flip:    bool = True,
        in_frequency_domain: bool = False,
    ) -> "ImagePropertiesEmphasizer":
        if in_frequency_domain:
            return self._apply_in_frequency_domain(
                lambda img: img.shift(shift_x, shift_y, cyclic=cyclic, flip=flip)
            )

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
        freq_u:    float,
        freq_v:    float,
        in_frequency_domain: bool = False,
    ) -> "ImagePropertiesEmphasizer":
        if in_frequency_domain:
            return self._apply_in_frequency_domain(
                lambda img: img.multiply_by_complex_exponential(amplitude, freq_u, freq_v)
            )

        arr = self._array
        H, W = arr.shape
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        kernel = amplitude * np.exp(1j * 2 * np.pi * (freq_u * xx + freq_v * yy))
        return self._wrap(arr * kernel)

    def stretch(
        self,
        stretch_x: float,
        stretch_y: float,
        order:     int  = 3,
        in_frequency_domain: bool = False,
    ) -> "ImagePropertiesEmphasizer":
        if in_frequency_domain:
            return self._apply_in_frequency_domain(
                lambda img: img.stretch(stretch_x, stretch_y, order=order)
            )

        arr     = self._array
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
        in_frequency_domain: bool = False,
    ) -> "ImagePropertiesEmphasizer":
        if in_frequency_domain:
            return self._apply_in_frequency_domain(
                lambda img: img.mirror(mirror_axis, duplicate_mode)
            )

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
        in_frequency_domain: bool = False,
    ) -> "ImagePropertiesEmphasizer":
        if in_frequency_domain:
            return self._apply_in_frequency_domain(
                lambda img: img.make_even_or_odd(symmetry_type)
            )

        arr = self._coerce(self._array)
        if np.iscomplexobj(arr):
            img_centered, mean = arr, 0
        else:
            mean         = np.mean(arr)
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
        order: int  = 3,
        in_frequency_domain: bool = False,
    ) -> "ImagePropertiesEmphasizer":
        if in_frequency_domain:
            return self._apply_in_frequency_domain(
                lambda img: img.rotate(angle, order=order)
            )

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
        in_frequency_domain: bool = False,
    ) -> "ImagePropertiesEmphasizer":
        if in_frequency_domain:
            return self._apply_in_frequency_domain(
                lambda img: img.differentiate(axis, method=method)
            )

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

    def integrate(
        self,
        axis: str = "x",
        in_frequency_domain: bool = False,
    ) -> "ImagePropertiesEmphasizer":
        if in_frequency_domain:
            return self._apply_in_frequency_domain(
                lambda img: img.integrate(axis)
            )

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
        in_frequency_domain: bool = False,
        **kwargs,
    ) -> "ImagePropertiesEmphasizer":
        if in_frequency_domain:
            return self._apply_in_frequency_domain(
                lambda img: img.multiply_by_window(
                    window_width, window_height, center_x, center_y,
                    window_type=window_type, **kwargs,
                )
            )

        arr        = self._array
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
            y     = np.arange(window_height) - window_height / 2
            x     = np.arange(window_width)  - window_width  / 2
            win_y = np.exp(-(y ** 2) / (2 * sig_y ** 2))
            win_x = np.exp(-(x ** 2) / (2 * sig_x ** 2))
        else:
            raise ValueError(f"Unsupported window type: {window_type}")

        window_2d = np.outer(win_y, win_x)
        padded    = np.zeros((rows, cols), dtype=np.float64)
        padded[start_y:end_y, start_x:end_x] = window_2d

        return self._wrap(arr * padded)

    # ──────────────────────────────────────────────────────────────────────
    # Complex-spatial display helper
    # ──────────────────────────────────────────────────────────────────────

    def get_complex_spatial_components(self) -> dict:
        arr   = self._array
        mag   = np.abs(arr)
        phase = np.angle(arr)
        real  = np.real(arr)
        imag  = np.imag(arr)

        def _minmax(a):
            lo, hi = a.min(), a.max()
            return (a - lo) / (hi - lo) if hi > lo else a

        return {
            "magnitude": _minmax(mag),
            "phase":     (phase + np.pi) / (2 * np.pi),
            "real":      _minmax(real),
            "imaginary": _minmax(imag),
        }
    
    def apply_n_ffts(
    self,
    n: int,
    in_frequency_domain: bool = False,
    ) -> "ImagePropertiesEmphasizer":
        """
        Applies the 2D FFT n times using the modulo-4 identity:
        n%4 == 0 → original
        n%4 == 1 → FFT
        n%4 == 2 → flipped (180° rotation)
        n%4 == 3 → flipped FFT
        """
        if in_frequency_domain:
            return self._apply_in_frequency_domain(
                lambda img: img.apply_n_ffts(n)
            )

        state = n % 4
        arr   = self._array

        if state == 0:
            return self._wrap(arr)
        elif state == 1:
            return self._wrap(np.fft.fft2(arr, norm='ortho'))
        elif state == 2:
            return self._wrap(arr[::-1, ::-1])
        elif state == 3:
            return self._wrap(np.fft.fft2(arr, norm='ortho')[::-1, ::-1])