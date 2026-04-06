import numpy as np
from PIL import Image
import io

def prepare_fft_for_display(F):
    """
    Takes a raw 2D FFT, centers the DC component, extracts all 4 components,
    and normalizes all of them to a 0.0 - 1.0 range for frontend visualization.
    
    Returns both shifted (DC-centered) and unshifted versions of all display images.
    """

    def _process_components(F_arr):
        """Extract and normalize all 4 components from a (possibly shifted) FFT array."""
        magnitude_spectrum = np.abs(F_arr)
        raw_phase         = np.angle(F_arr)
        raw_real          = np.real(F_arr)
        raw_imaginary     = np.imag(F_arr)

        scale = 100000.0

        # MAGNITUDE — log scale + max normalization
        mag_log = np.log(1 + (magnitude_spectrum * scale))
        mag_max = np.max(mag_log)
        display_magnitude = mag_log / mag_max if mag_max > 0 else mag_log

        # PHASE — [-pi, pi] → [0, 1]
        display_phase = (raw_phase + np.pi) / (2 * np.pi)

        # REAL — signed log + min-max normalization
        real_log = np.sign(raw_real) * np.log(1 + np.abs(raw_real * scale))
        real_min, real_max = np.min(real_log), np.max(real_log)
        display_real = (
            (real_log - real_min) / (real_max - real_min)
            if real_max > real_min else real_log
        )

        # IMAGINARY — signed log + min-max normalization
        imag_log = np.sign(raw_imaginary) * np.log(1 + np.abs(raw_imaginary * scale))
        imag_min, imag_max = np.min(imag_log), np.max(imag_log)
        display_imaginary = (
            (imag_log - imag_min) / (imag_max - imag_min)
            if imag_max > imag_min else imag_log
        )

        return display_magnitude, display_phase, display_real, display_imaginary

    # Shifted (DC-centered) versions
    F_shifted = np.fft.fftshift(F)
    shifted_mag, shifted_phase, shifted_real, shifted_imag = _process_components(F_shifted)

    # Unshifted versions — same processing pipeline, raw FFT layout
    unshifted_mag, unshifted_phase, unshifted_real, unshifted_imag = _process_components(F)

    return (
        F,                                                    # raw FFT for future math
        shifted_mag, shifted_phase, shifted_real, shifted_imag,       # DC-centered
        unshifted_mag, unshifted_phase, unshifted_real, unshifted_imag,  # raw layout
    )


def prepare_complex_spatial_for_display(img_complex):
    """
    Takes a spatial image that contains complex numbers, extracts all 4 components,
    and normalizes them to a 0.0 - 1.0 range for frontend visualization.
    """
    # 2. Extract the raw components
    magnitude = np.abs(img_complex)
    raw_phase = np.angle(img_complex)               
    raw_real = np.real(img_complex)
    raw_imaginary = np.imag(img_complex)
    
    # 3. Process MAGNITUDE (Standard Min-Max only. NO Log Scale needed!)
    # Fun fact: Because the magnitude of a complex exponential is exactly 1,
    # the magnitude image will just look exactly like your original, unmodified input image!
    mag_min, mag_max = np.min(magnitude), np.max(magnitude)
    if mag_max > mag_min:
        display_magnitude = (magnitude - mag_min) / (mag_max - mag_min)
    else:
        display_magnitude = magnitude
    
    # 4. Process PHASE (Normalize from [-pi, pi] to [0.0, 1.0])
    # This will look like a repeating gradient ramp showing the angle of the complex wave.
    display_phase = (raw_phase + np.pi) / (2 * np.pi)
    
    # 5. Process REAL and IMAGINARY (Standard Min-Max only. NO Log Scale needed!)
    # The Real part is the image multiplied by a Cosine wave.
    # The Imaginary part is the image multiplied by a Sine wave.
    # We just min-max them so the negative troughs of the wave become black, 
    # the positive peaks become white, and 0 becomes middle-gray.
    real_min, real_max = np.min(raw_real), np.max(raw_real)
    if real_max > real_min:
        display_real = (raw_real - real_min) / (real_max - real_min)
    else:
        display_real = raw_real
        
    imag_min, imag_max = np.min(raw_imaginary), np.max(raw_imaginary)
    if imag_max > imag_min:
        display_imaginary = (raw_imaginary - imag_min) / (imag_max - imag_min)
    else:
        display_imaginary = raw_imaginary

    # Return the raw complex spatial array for future math, plus the 4 display images
    return img_complex, display_magnitude, display_phase, display_real, display_imaginary

def arr_to_bytes(arr: np.ndarray, fmt="JPEG") -> bytes:
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=95)
    return buf.getvalue()

def MultipleFourierTransforms(img, n, scenario_type):
    """
    Applies the Fourier Transform n times based on the scenario type.
    
    scenario_type: 
      'A' -> Started as a normal Spatial Image
      'B' -> Started as a Frequency Domain (FFT) array
      'C' -> Started as a Complex Spatial Image (Multiplied by complex exponential)
      
    Returns:
    tuple: (finalOutput_Math, Magnitude_img, Phase_img, Real_img, Imaginary_img)
    """
    
    # ==========================================
    # 1. APPLY THE MATH (Modulo 4 trick)
    # ==========================================
    state = n % 4
    
    if state == 0:
        raw_output = img
    elif state == 1:
        raw_output = np.fft.fft2(img, norm='ortho')
    elif state == 2:
        raw_output = img[::-1, ::-1]
    elif state == 3:
        F = np.fft.fft2(img, norm='ortho')
        raw_output = F[::-1, ::-1]

    # ==========================================
    # 2. ROUTE THE DISPLAY LOGIC
    # ==========================================
    # Determine if we should treat the current output as a complex/frequency 
    # spectrum that needs shifting and log-scaling.
    use_fft_display = False
    
    if scenario_type == 'A' and state in [1, 3]:
        use_fft_display = True
    elif scenario_type == 'B' and state in [0, 2]:
        use_fft_display = True
    elif scenario_type == 'C':
        # Scenario C is ALWAYS complex, so we treat it like an FFT for the UI
        # so it gets broken down properly into the 4 components.
        use_fft_display = True

    # ==========================================
    # 3. RETURN FORMATTED UI VARIABLES
    # ==========================================
    if use_fft_display:
        # Pass it through your dedicated FFT preparation function
        return prepare_fft_for_display(raw_output)
        
    else:
        # It's a standard purely Real spatial image. 
        # Bypassing prepare_fft_for_display() to prevent shifting and log-scaling.
        
        # Clean up microscopic floating point math noise (e.g. + 1e-15j)
        cleaned_output = np.real(raw_output)
        
        # Normalize the image cleanly between 0.0 and 1.0 for the UI
        img_min, img_max = np.min(cleaned_output), np.max(cleaned_output)
        if img_max > img_min:
            norm_img = (cleaned_output - img_min) / (img_max - img_min)
        else:
            norm_img = np.zeros_like(cleaned_output)
            
        # For a standard spatial image: 
        # Magnitude and Real are just the image. Phase and Imaginary are empty (black).
        display_mag = norm_img
        display_real = norm_img
        display_phase = np.zeros_like(cleaned_output)
        display_imag = np.zeros_like(cleaned_output)
        
        return cleaned_output, display_mag, display_phase, display_real, display_imag, display_mag, display_phase, display_real, display_imag
    

def ReconstructImageFromFFT(F_complex, was_shifted=False, preserve_complex=False):
    """
    Takes a 2D complex FFT array and reconstructs the spatial image.
    
    Parameters:
    F_complex (2D numpy array): The frequency domain data.
    was_shifted (bool): Set to True if the DC component is currently in the center.
    """
    # 1. Undo the shift if the frequency origin is currently in the center
    if was_shifted:
        F_ready = np.fft.ifftshift(F_complex)
    else:
        F_ready = F_complex
        
    # 2. Perform the 2D Inverse Fast Fourier Transform
    img_reconstructed = np.fft.ifft2(F_ready)

    if preserve_complex:
        return img_reconstructed
    
    # 3. Discard microscopic imaginary rounding errors left by the math engine
    img_real = np.real(img_reconstructed)
    
    # 4. Normalize the data to span the full 0-255 range
    img_min = np.min(img_real)
    img_max = np.max(img_real)
    
    if img_max > img_min:
        img_normalized = (img_real - img_min) / (img_max - img_min)
    else:
        # Fallback in case of a completely flat/blank image
        img_normalized = np.zeros_like(img_real)
    
    return img_normalized
