import os
import base64
import numpy as np
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image
import io
import json

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from Services.ImageProcessingService import (
    RotateImage, ShiftImage, MultiplyImageByComplexExponential,
    StretchImage, MirrorImage, MakeImageEvenOrOdd,
    DifferentiateImage, IntegrateImage, MultiplyByWindow
)
from Services.FrequencyTransformService import (
    MultipleFourierTransforms, ReconstructImageFromFFT, prepare_fft_for_display,
    prepare_complex_spatial_for_display, arr_to_bytes
)

app = FastAPI()
origins = ["*"]
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# complex_input: JSON string: {real, imag, shape}
async def parse_image_input(image: Optional[UploadFile],complex_input: Optional[str]) -> np.ndarray:
    if complex_input is not None:
        try:
            data = json.loads(complex_input)
            shape = tuple(data["shape"])
            real_part = np.array(data["real"], dtype=np.float32).reshape(shape)

            if data.get("imag") is not None:
                imag_part = np.array(data["imag"], dtype=np.float32).reshape(shape)
                return (real_part + 1j * imag_part).astype(np.complex64)
            else:
                return real_part  # purely real, stored as float32

        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Malformed complex_input JSON: {exc}"
            )
        
    if image is not None:
        img_data = await image.read()
        pil_image = Image.open(io.BytesIO(img_data)).convert("L")
        return np.array(pil_image, dtype=np.float32) / 255.0

    raise HTTPException(
        status_code=400,
        detail="Provide either 'image' (file upload) or 'complex_input' (JSON)."
    )
import io
import json
import numpy as np
from PIL import Image
from fastapi.responses import StreamingResponse

def stream_array(arr: np.ndarray) -> StreamingResponse:
    """
    Stream a numpy array. 
    - If real: returns a single JPEG image.
    - If complex: uses `prepare_fft_for_display` to process components, 
      then returns a multipart/form-data response with JSON metadata 
      and 4 JPEG representations (magnitude, phase, real, imaginary).
    """
    # ==========================================
    # SCENARIO 1: REAL ARRAY (Single Image)
    # ==========================================
    if not np.iscomplexobj(arr):
        arr_uint8 = arr if arr.dtype == np.uint8 else np.clip(arr * 255, 0, 255).astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(arr_uint8).save(buffer, format="JPEG", quality=95)
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="image/jpeg")

    # ==========================================
    # SCENARIO 2: COMPLEX ARRAY (Multipart FFT)
    # ==========================================
    boundary = "fourier_boundary"
    def arr_to_bytes(a: np.ndarray) -> bytes:
        """Converts a 0.0-1.0 float array directly to JPEG bytes."""
        a_uint8 = np.clip(a * 255.0, 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(a_uint8).save(buf, format="JPEG", quality=95)
        return buf.getvalue()

    def build_part(name: str, content: bytes, content_type: str) -> bytes:
        header = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8")
        return header + content + b"\r\n"

    # --- 1. Process Complex Array ---
    F_raw, mag_disp, phase_disp, real_disp, imag_disp = prepare_fft_for_display(arr)

    # --- 3. Convert 4 normalized display images to JPEG bytes ---
    mag_bytes   = arr_to_bytes(mag_disp)
    phase_bytes = arr_to_bytes(phase_disp)
    real_bytes  = arr_to_bytes(real_disp)
    imag_bytes  = arr_to_bytes(imag_disp)

    # --- 4. Build multipart response ---
    body = b""
    body += build_part("magnitude", mag_bytes,                     "image/jpeg")
    body += build_part("phase",     phase_bytes,                   "image/jpeg")
    body += build_part("real",      real_bytes,                    "image/jpeg")
    body += build_part("imaginary", imag_bytes,                    "image/jpeg")
    body += f"--{boundary}--\r\n".encode("utf-8")

    return StreamingResponse(
        io.BytesIO(body),
        media_type=f"multipart/form-data; boundary={boundary}",
    )

def build_multipart_response_for_complex_exponential(img_complex: np.ndarray, boundary: str) -> StreamingResponse:
    """Serialize a complex (or real) array + 4 display images as multipart."""
    _, display_magnitude, display_phase, display_real, display_imaginary = \
        prepare_complex_spatial_for_display(img_complex)

    def build_part(name: str, content: bytes, content_type: str) -> bytes:
        header = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode()
        return header + content + b"\r\n"

    body = b""
    body += build_part("magnitude", arr_to_bytes(display_magnitude),  "image/jpeg")
    body += build_part("phase",     arr_to_bytes(display_phase),      "image/jpeg")
    body += build_part("real",      arr_to_bytes(display_real),       "image/jpeg")
    body += build_part("imaginary", arr_to_bytes(display_imaginary),  "image/jpeg")
    body += f"--{boundary}--\r\n".encode()

    return StreamingResponse(
        io.BytesIO(body),
        media_type=f"multipart/form-data; boundary={boundary}",
    )

@app.post("/shiftimage")
async def ShiftImageEndpoint(
    shift_x: int   = Form(...),
    shift_y: int   = Form(...),
    cyclic:  bool  = Form(...),
    flip:    bool  = Form(...),
    image:         Optional[UploadFile] = File(None),
    complex_input: Optional[str]        = Form(None),
):
    arr = await parse_image_input(image, complex_input)
    result = ShiftImage(arr, shift_x, shift_y, cyclic=cyclic, flip=flip)
    return stream_array(result)


@app.post("/multiplybycomplexexponential")
async def MultiplyByComplexExponentialEndpoint(
    amplitude: float = Form(...),
    freq_u:    float = Form(...),
    freq_v:    float = Form(...),
    image:         Optional[UploadFile] = File(None),
    complex_input: Optional[str]        = Form(None),
):
    arr = await parse_image_input(image, complex_input)
    result = MultiplyImageByComplexExponential(arr, amplitude, freq_u, freq_v)
    return build_multipart_response_for_complex_exponential(result, boundary="complex_exponential_boundary")


@app.post("/stretchimage")
async def StretchImageEndpoint(
    stretch_x: float = Form(...),
    stretch_y: float = Form(...),
    image:         Optional[UploadFile] = File(None),
    complex_input: Optional[str]        = Form(None),
):
    arr = await parse_image_input(image, complex_input)
    result = StretchImage(arr, stretch_x, stretch_y)
    return stream_array(result)


@app.post("/mirrorimage")
async def MirrorImageEndpoint(
    mirror_axis:    str  = Form(...),
    duplicate_mode: bool = Form(...),
    image:         Optional[UploadFile] = File(None),
    complex_input: Optional[str]        = Form(None),
):
    arr = await parse_image_input(image, complex_input)
    result = MirrorImage(arr, mirror_axis, duplicate_mode)
    return stream_array(result)


@app.post("/makeimageevenorodd")
async def MakeImageEvenOrOddEndpoint(
    symmetry_type: str = Form(...),
    image:         Optional[UploadFile] = File(None),
    complex_input: Optional[str]        = Form(None),
):
    arr = await parse_image_input(image, complex_input)
    result = MakeImageEvenOrOdd(arr, symmetry_type)
    return stream_array(result)


@app.post("/rotateimage")
async def RotateImageEndpoint(
    angle: float = Form(...),
    image:         Optional[UploadFile] = File(None),
    complex_input: Optional[str]        = Form(None),
):
    arr = await parse_image_input(image, complex_input)
    result = RotateImage(arr, angle)
    return stream_array(result)


@app.post("/differentiateimage")
async def DifferentiateImageEndpoint(
    axis: str = Form(...),
    image:         Optional[UploadFile] = File(None),
    complex_input: Optional[str]        = Form(None),
):
    arr = await parse_image_input(image, complex_input)
    result = DifferentiateImage(arr, axis)
    return stream_array(result)


@app.post("/integrateimage")
async def IntegrateImageEndpoint(
    axis: str = Form(...),
    image:         Optional[UploadFile] = File(None),
    complex_input: Optional[str]        = Form(None),
):
    arr = await parse_image_input(image, complex_input)
    result = IntegrateImage(arr, axis)
    return stream_array(result)


@app.post("/multiplybywindow")
async def MultiplyByWindowEndpoint(
    window_type: str = Form(...),
    sigma_x: Optional[float] = Query(None),
    sigma_y: Optional[float] = Query(None),
    image:         Optional[UploadFile] = File(None),
    complex_input: Optional[str]        = Form(None),
):
    arr = await parse_image_input(image, complex_input)
    kwargs = {}
    if sigma_x is not None: kwargs["sigma_x"] = sigma_x
    if sigma_y is not None: kwargs["sigma_y"] = sigma_y
    windowed, _ = MultiplyByWindow(arr, window_type=window_type, **kwargs)
    return stream_array(windowed)

@app.post("/fft")
async def FFTEndpoint(
    scenario_type: str = Form(..., description="A | B | C"),
    n: int = Query(..., description="Number of Fourier Transforms to apply"),
    image: Optional[UploadFile] = File(None),
    complex_input: Optional[str] = Form(None),
):
    # --- Load image OR complex array ---
    # This replaces the manual `image.read()` and PIL conversion,
    # routing either the uploaded file or the complex JSON string into `arr`
    arr = await parse_image_input(image, complex_input)

    # --- Run function ---
    final_output, mag, phase, real, imag = MultipleFourierTransforms(
        arr, n=n, scenario_type=scenario_type
    )

    # --- Convert 4 display images to JPEG bytes ---
    # (Assuming arr_to_bytes is defined globally or imported)
    mag_bytes   = arr_to_bytes(mag)
    phase_bytes = arr_to_bytes(phase)
    real_bytes  = arr_to_bytes(real)
    imag_bytes  = arr_to_bytes(imag)

    # --- Build multipart response ---
    boundary = "fourier_boundary"
    
    def build_part(name: str, content: bytes, content_type: str) -> bytes:
        header = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8")
        return header + content + b"\r\n"

    body = b""
    body += build_part("magnitude", mag_bytes,                         "image/jpeg")
    body += build_part("phase",     phase_bytes,                       "image/jpeg")
    body += build_part("real",      real_bytes,                        "image/jpeg")
    body += build_part("imaginary", imag_bytes,                        "image/jpeg")
    body += f"--{boundary}--\r\n".encode("utf-8")

    return StreamingResponse(
        io.BytesIO(body),
        media_type=f"multipart/form-data; boundary={boundary}",
    )

@app.post("/ifft")
async def IFFTEndpoint(
    wasShifted: bool = Form(...),
    image:         Optional[UploadFile] = File(None),
    complex_input: Optional[str]        = Form(None),
):
    arr = await parse_image_input(image, complex_input)
    result = ReconstructImageFromFFT(arr, wasShifted)
    return stream_array(result)

@app.post("/fft_then_operate")
async def FftThenOperateEndpoint(
    action: str = Form(...),
    # Shift
    shift_x: int = Form(0),
    shift_y: int = Form(0),
    cyclic:  bool = Form(True),
    flip:    bool = Form(False),
    # Stretch
    stretch_x: float = Form(1.0),
    stretch_y: float = Form(1.0),
    # Rotate
    angle: float = Form(0.0),
    # Mirror
    mirror_axis:    str  = Form("horizontal"),
    duplicate_mode: bool = Form(False),
    # Complex exponential
    amplitude: float = Form(1.0),
    freq_u:    float = Form(0.0),
    freq_v:    float = Form(0.0),
    # Window
    window_type: str   = Form("gaussian"),
    sigma_x:     float = Form(30.0),
    sigma_y:     float = Form(30.0),
    # Symmetry / calculus
    symmetry_type: str = Form("even"),
    axis:          str = Form("x"),
    # FT repeat
    scenario_type: str = Form("B"),
    n:             int = Form(1),
    # Image input
    image: Optional[UploadFile] = File(None),
    complex_input: Optional[str] = Form(None),
):
    # 1. Parse input
    arr = await parse_image_input(image, complex_input)

    # 2. FFT the input first
    fft_result, _, _, _, _ = MultipleFourierTransforms(arr, n=1, scenario_type="A")

    # 3. Apply the chosen operation on the FFT result
    if action == "shift":
        operated = ShiftImage(fft_result, shift_x, shift_y, cyclic=cyclic, flip=flip)
    elif action == "stretch":
        operated = StretchImage(fft_result, stretch_x, stretch_y)
    elif action == "rotate":
        operated = RotateImage(fft_result, angle)
    elif action == "mirror":
        operated = MirrorImage(fft_result, mirror_axis, duplicate_mode)
    elif action == "even":
        operated = MakeImageEvenOrOdd(fft_result, "even")
    elif action == "odd":
        operated = MakeImageEvenOrOdd(fft_result, "odd")
    elif action == "complex_exp":
        operated = MultiplyImageByComplexExponential(fft_result, 1, freq_u, freq_v)
    elif action == "window":
        operated, _ = MultiplyByWindow(fft_result, window_type=window_type, sigma_x=sigma_x, sigma_y=sigma_y)
    elif action == "differentiate":
        operated = DifferentiateImage(fft_result, axis)
    elif action == "integrate":
        operated = IntegrateImage(fft_result, axis)
    elif action == "ft_repeat":
        operated, _, _, _, _ = MultipleFourierTransforms(fft_result, n=n, scenario_type=scenario_type)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")

    # 4. IFFT back to spatial domain
    spatial_result = ReconstructImageFromFFT(operated, was_shifted=False)
    spatial_bytes = arr_to_bytes(
        np.clip(np.real(spatial_result), 0, 1).astype(np.float32)
        if np.iscomplexobj(spatial_result)
        else np.clip(spatial_result, 0, 1).astype(np.float32)
    )
 
    # 5. Also compute FFT of the operated result for the frequency display
    _, mag, phase, real, imag = prepare_fft_for_display(operated)

    # 6. Return multipart: spatial image + 4 FFT display images + metadata
    boundary = "fft_operate_boundary"

    def build_part(name: str, content: bytes, content_type: str) -> bytes:
        header = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8")
        return header + content + b"\r\n"

    body = b""
    body += build_part("spatial",    spatial_bytes,    "image/jpeg")
    body += build_part("magnitude",  arr_to_bytes(mag),   "image/jpeg")
    body += build_part("phase",      arr_to_bytes(phase), "image/jpeg")
    body += build_part("real",       arr_to_bytes(real),  "image/jpeg")
    body += build_part("imaginary",  arr_to_bytes(imag),  "image/jpeg")
    body += f"--{boundary}--\r\n".encode("utf-8")

    return StreamingResponse(
        io.BytesIO(body),
        media_type=f"multipart/form-data; boundary={boundary}",
    )