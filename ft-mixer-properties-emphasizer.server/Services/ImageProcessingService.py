import numpy as np
from scipy.ndimage import zoom, rotate, convolve

def ShiftImage(image, shift_x=0, shift_y=0, cyclic=False, flip=True):
    img_array = np.asarray(image)
    
    if flip:
        shift_y = -shift_y
        
    shifted = np.roll(img_array, shift=shift_y, axis=0)
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
            
    return shifted


def MultiplyImageByComplexExponential(image, amplitude: float, freq_u: float, freq_v: float):
    img_array = np.asarray(image)
    H, W = img_array.shape
    
    x = np.arange(W)
    y = np.arange(H)
    xx, yy = np.meshgrid(x, y)
    
    exp_kernel = amplitude * np.exp(1j * 2 * np.pi * (freq_u * xx + freq_v * yy))
    
    return img_array * exp_kernel


def StretchImage(image, stretch_x: float, stretch_y: float, order: int = 3):
    img_array = np.asarray(image)
    zoom_factors = (stretch_y, stretch_x)
    
    if np.iscomplexobj(img_array):
        real_stretched = zoom(img_array.real, zoom_factors, order=order)
        imag_stretched = zoom(img_array.imag, zoom_factors, order=order)
        return real_stretched + 1j * imag_stretched
    else:
        return zoom(img_array, zoom_factors, order=order)


def MirrorImage(image, mirror_axis: str, duplicate_mode: bool = True):
    img_array = np.asarray(image)
    
    if not duplicate_mode:
        if mirror_axis == 'horizontal':
            return np.flipud(img_array)
        elif mirror_axis == 'vertical':
            return np.fliplr(img_array)
        else:
            return np.flipud(np.fliplr(img_array))
    else:
        if mirror_axis == 'horizontal':
            return np.vstack([img_array, np.flipud(img_array)])
        elif mirror_axis == 'vertical':
            return np.hstack([img_array, np.fliplr(img_array)])
        else:
            top = np.hstack([img_array, np.fliplr(img_array)])
            bottom = np.hstack([np.flipud(img_array), np.flipud(np.fliplr(img_array))])
            return np.vstack([top, bottom])


def MakeImageEvenOrOdd(image, symmetry_type: str):
    img_array = np.asarray(image)
    f_neg = np.rot90(img_array, k=2)
    
    if symmetry_type == 'even':
        return 0.5 * (img_array + f_neg)
    elif symmetry_type == 'odd':
        raw_odd = 0.5 * (img_array - f_neg)
        
        # Min-Max Normalization to map 0 to mid-gray
        o_min = np.min(raw_odd)
        o_max = np.max(raw_odd)
        
        if o_max > o_min:
            normalized_odd = (raw_odd - o_min) / (o_max - o_min) * 255.0
        else:
            normalized_odd = raw_odd
            
        return normalized_odd.astype(np.uint8)
    else:
        raise ValueError("symmetry_type must be 'even' or 'odd'")


def RotateImage(image, angle: float, order: int = 3):
    img_array = np.asarray(image)
    
    if np.iscomplexobj(img_array):
        real_rot = rotate(img_array.real, angle, reshape=True, order=order)
        imag_rot = rotate(img_array.imag, angle, reshape=True, order=order)
        return real_rot + 1j * imag_rot
    else:
        return rotate(img_array, angle, reshape=True, order=order)


def DifferentiateImage(image, axis='x', method='central'):
    img_array = np.asarray(image)
    
    if method == 'central':
        if axis == 'x':
            kernel = np.array([[1, 0, -1]]) / 2.0
        elif axis == 'y':
            kernel = np.array([[1], [0], [-1]]) / 2.0
            
    elif method == 'sobel':
        if axis == 'x':
            kernel = np.array([[-1, 0, 1], 
                               [-2, 0, 2], 
                               [-1, 0, 1]]) / 8.0
        elif axis == 'y':
            kernel = np.array([[-1, -2, -1], 
                               [ 0,  0,  0], 
                               [ 1,  2,  1]]) / 8.0
    else:
        raise ValueError("Method must be 'central' or 'sobel'")

    raw_derivative = convolve(img_array, kernel, mode='reflect')
    
    # Normalize to a 0-255 range so 0 becomes middle-gray
    d_min = np.min(raw_derivative)
    d_max = np.max(raw_derivative)
    
    if d_max > d_min:
        normalized = (raw_derivative - d_min) / (d_max - d_min) * 255.0
    else:
        normalized = raw_derivative
        
    return normalized.astype(np.uint8)

def IntegrateImage(image, axis='x'):
    img_array = np.asarray(image)
    
    if axis == 'x':
        return np.cumsum(img_array, axis=1)
    elif axis == 'y':
        return np.cumsum(img_array, axis=0)
    else:
        raise ValueError("Axis must be 'x' or 'y'")


def MultiplyByWindow(image, window_width, window_height, center_x, center_y, window_type='hamming', **kwargs):
    img_array = np.asarray(image)
    rows, cols = img_array.shape
    
    # Calculate the starting and ending indices for the window placement
    start_y = center_y - (window_height // 2)
    end_y = start_y + window_height
    
    start_x = center_x - (window_width // 2)
    end_x = start_x + window_width
    
    # Check if the window bounds fall outside the image dimensions
    if start_y < 0 or end_y > rows or start_x < 0 or end_x > cols:
        raise ValueError(
            f"Window placement is impossible. A ({window_width}x{window_height}) window "
            f"centered at ({center_x}, {center_y}) exceeds the image bounds of ({cols}x{rows})."
        )
    
    # Generate the 1D windows using the provided width and height
    if window_type == 'rectangular':
        win_y = np.ones(window_height)
        win_x = np.ones(window_width)
        
    elif window_type == 'hamming':
        win_y = np.hamming(window_height)
        win_x = np.hamming(window_width)
        
    elif window_type == 'hanning':
        win_y = np.hanning(window_height)
        win_x = np.hanning(window_width)
        
    elif window_type == 'gaussian':
        sigma_y = kwargs.get('sigma_y', window_height / 6)
        sigma_x = kwargs.get('sigma_x', window_width / 6)
        y = np.arange(window_height) - window_height / 2
        x = np.arange(window_width) - window_width / 2
        win_y = np.exp(-(y**2) / (2 * sigma_y**2))
        win_x = np.exp(-(x**2) / (2 * sigma_x**2))
        
    else:
        raise ValueError("Unsupported window type.")

    # Create the 2D window of size (window_height, window_width)
    window_2d = np.outer(win_y, win_x)
    
    # Create a zero matrix matching the original image dimensions
    padded_window = np.zeros((rows, cols))
    
    # Embed the 2D window into the zero matrix at the specified coordinates
    padded_window[start_y:end_y, start_x:end_x] = window_2d
    
    # Return the multiplied image and the padded window
    return img_array * padded_window





# ══════════════════════════════════════════════════════════════════════════════
# Part A helpers  (new — used by MixerService)
# ══════════════════════════════════════════════════════════════════════════════

def compute_target_size(
    sizes: list[tuple[int, int]],
    policy: str,
) -> tuple[int, int]:
    """
    Compute the unified target (width, height) from a list of image sizes.

    policy:
      'smallest' → pick the size with the fewest total pixels
      'largest'  → pick the size with the most total pixels
      'fixed'    → use the first image's size as the reference
    """
    if not sizes:
        return (256, 256)

    if policy == "smallest":
        return min(sizes, key=lambda s: s[0] * s[1])
    elif policy == "largest":
        return max(sizes, key=lambda s: s[0] * s[1])
    else:                          # 'fixed'
        return sizes[0]


def load_and_unify_images(
    images_bytes: list[bytes],
    unify_policy: str,
    keep_aspect_ratio: bool,
) -> list[np.ndarray]:
    """
    Decode raw image bytes, convert to grayscale, resize to a unified size.
    Returns a list of float64 numpy arrays in the range 0–255.
    """
    pil_images = [
        Image.open(io.BytesIO(b)).convert("L")
        for b in images_bytes
    ]

    target_size = compute_target_size(
        [img.size for img in pil_images], unify_policy
    )

    resample = Image.LANCZOS
    result   = []

    for img in pil_images:
        if keep_aspect_ratio:
            img.thumbnail(target_size, resample)
            # Pad to exact target size (center the image on a black canvas)
            canvas = Image.new("L", target_size, 0)
            offset = (
                (target_size[0] - img.width)  // 2,
                (target_size[1] - img.height) // 2,
            )
            canvas.paste(img, offset)
            result.append(np.array(canvas, dtype=np.float64))
        else:
            result.append(
                np.array(img.resize(target_size, resample), dtype=np.float64)
            )

    return result




