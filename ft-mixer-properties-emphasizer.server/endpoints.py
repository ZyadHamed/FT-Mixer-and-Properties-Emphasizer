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
from Services.MixerService import MixerService

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

mixer_service = MixerService()


# ── Shared helpers ────────────────────────────────────────────────────────────

async def parse_image_input(image: Optional[UploadFile], complex_input: Optional[str]) -> np.ndarray:
    if complex_input is not None:
        try:
            data = json.loads(complex_input)
            shape = tuple(data["shape"])
            real_part = np.array(data["real"], dtype=np.float32).reshape(shape)

            if data.get("imag") is not None:
                imag_part = np.array(data["imag"], dtype=np.float32).reshape(shape)
                return (real_part + 1j * imag_part).astype(np.complex64)
            else:
                return real_part

        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=f"Malformed complex_input JSON: {exc}")

    if image is not None:
        img_data = await image.read()
        pil_image = Image.open(io.BytesIO(img_data)).convert("L")
        return np.array(pil_image, dtype=np.float32) / 255.0

    raise HTTPException(
        status_code=400,
        detail="Provide either 'image' (file upload) or 'complex_input' (JSON)."
    )


def stream_array(arr: np.ndarray) -> StreamingResponse:
    if not np.iscomplexobj(arr):
        arr_uint8 = arr if arr.dtype == np.uint8 else np.clip(arr * 255, 0, 255).astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(arr_uint8).save(buffer, format="JPEG", quality=95)
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="image/jpeg")

    boundary = "fourier_boundary"

    def _arr_to_bytes(a: np.ndarray) -> bytes:
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

    F_raw, mag_disp, phase_disp, real_disp, imag_disp = prepare_fft_for_display(arr)

    body = b""
    body += build_part("magnitude", _arr_to_bytes(mag_disp),   "image/jpeg")
    body += build_part("phase",     _arr_to_bytes(phase_disp), "image/jpeg")
    body += build_part("real",      _arr_to_bytes(real_disp),  "image/jpeg")
    body += build_part("imaginary", _arr_to_bytes(imag_disp),  "image/jpeg")
    body += f"--{boundary}--\r\n".encode("utf-8")

    return StreamingResponse(
        io.BytesIO(body),
        media_type=f"multipart/form-data; boundary={boundary}",
    )


def build_multipart_response_for_complex_exponential(img_complex: np.ndarray, boundary: str) -> StreamingResponse:
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


# ══════════════════════════════════════════════════════════════════════════════
# Part B – FT Properties Emphasizer  (existing endpoints, unchanged)
# ══════════════════════════════════════════════════════════════════════════════

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
    arr = await parse_image_input(image, complex_input)
    final_output, mag, phase, real, imag = MultipleFourierTransforms(
        arr, n=n, scenario_type=scenario_type
    )

    mag_bytes   = arr_to_bytes(mag)
    phase_bytes = arr_to_bytes(phase)
    real_bytes  = arr_to_bytes(real)
    imag_bytes  = arr_to_bytes(imag)

    boundary = "fourier_boundary"

    def build_part(name: str, content: bytes, content_type: str) -> bytes:
        header = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8")
        return header + content + b"\r\n"

    body = b""
    body += build_part("magnitude", mag_bytes,   "image/jpeg")
    body += build_part("phase",     phase_bytes, "image/jpeg")
    body += build_part("real",      real_bytes,  "image/jpeg")
    body += build_part("imaginary", imag_bytes,  "image/jpeg")
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
    shift_x: int = Form(0),
    shift_y: int = Form(0),
    cyclic:  bool = Form(True),
    flip:    bool = Form(False),
    stretch_x: float = Form(1.0),
    stretch_y: float = Form(1.0),
    angle: float = Form(0.0),
    mirror_axis:    str  = Form("horizontal"),
    duplicate_mode: bool = Form(False),
    amplitude: float = Form(1.0),
    freq_u:    float = Form(0.0),
    freq_v:    float = Form(0.0),
    window_type: str   = Form("gaussian"),
    sigma_x:     float = Form(30.0),
    sigma_y:     float = Form(30.0),
    symmetry_type: str = Form("even"),
    axis:          str = Form("x"),
    scenario_type: str = Form("B"),
    n:             int = Form(1),
    image: Optional[UploadFile] = File(None),
    complex_input: Optional[str] = Form(None),
):
    arr = await parse_image_input(image, complex_input)
    fft_result, _, _, _, _ = MultipleFourierTransforms(arr, n=1, scenario_type="A")

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

    spatial_result = ReconstructImageFromFFT(operated, was_shifted=False)
    spatial_bytes = arr_to_bytes(
        np.clip(np.real(spatial_result), 0, 1).astype(np.float32)
        if np.iscomplexobj(spatial_result)
        else np.clip(spatial_result, 0, 1).astype(np.float32)
    )

    _, mag, phase, real, imag = prepare_fft_for_display(operated)

    boundary = "fft_operate_boundary"

    def build_part(name: str, content: bytes, content_type: str) -> bytes:
        header = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8")
        return header + content + b"\r\n"

    body = b""
    body += build_part("spatial",   spatial_bytes,       "image/jpeg")
    body += build_part("magnitude", arr_to_bytes(mag),   "image/jpeg")
    body += build_part("phase",     arr_to_bytes(phase), "image/jpeg")
    body += build_part("real",      arr_to_bytes(real),  "image/jpeg")
    body += build_part("imaginary", arr_to_bytes(imag),  "image/jpeg")
    body += f"--{boundary}--\r\n".encode("utf-8")

    return StreamingResponse(
        io.BytesIO(body),
        media_type=f"multipart/form-data; boundary={boundary}",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Part A – FT Mixer  (new endpoints)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/image/upload")
async def UploadImageEndpoint(
    image:           UploadFile = File(...),
    unify_policy:    str  = Form("smallest"),
    keep_aspect_ratio: bool = Form(True),
):
    """
    Accepts a single image.
    Returns original + all 4 FT component images as base64 JPEGs.
    """
    image_bytes = await image.read()

    # Convert to grayscale numpy array (0–255 uint8)
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("L")
    arr = np.array(pil_img, dtype=np.float32) / 255.0   # 0.0–1.0

    # Compute FFT components
    fft        = np.fft.fft2(arr)
    fft_shift  = np.fft.fftshift(fft)

    _, mag_disp, phase_disp, real_disp, imag_disp = prepare_fft_for_display(fft_shift)

    def to_b64(a: np.ndarray) -> str:
        uint8 = np.clip(a * 255, 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(uint8).save(buf, format="JPEG", quality=95)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # Original image as base64
    orig_uint8 = np.clip(arr * 255, 0, 255).astype(np.uint8)
    orig_buf = io.BytesIO()
    Image.fromarray(orig_uint8).save(orig_buf, format="JPEG", quality=95)
    orig_b64 = base64.b64encode(orig_buf.getvalue()).decode("utf-8")

    return JSONResponse({
        "original":  orig_b64,
        "magnitude": to_b64(mag_disp),
        "phase":     to_b64(phase_disp),
        "real":      to_b64(real_disp),
        "imaginary": to_b64(imag_disp),
    })


@app.post("/api/mixer/run")
async def RunMixEndpoint(
    image_0: Optional[UploadFile] = File(None),
    image_1: Optional[UploadFile] = File(None),
    image_2: Optional[UploadFile] = File(None),
    image_3: Optional[UploadFile] = File(None),
    mag_weight_0:   float = Form(1.0),
    mag_weight_1:   float = Form(1.0),
    mag_weight_2:   float = Form(1.0),
    mag_weight_3:   float = Form(1.0),
    phase_weight_0: float = Form(1.0),
    phase_weight_1: float = Form(1.0),
    phase_weight_2: float = Form(1.0),
    phase_weight_3: float = Form(1.0),
    component_pair:    str   = Form("mag-phase"),
    region_type:       str   = Form("inner"),
    region_size:       float = Form(40.0),
    unify_policy:      str   = Form("smallest"),
    keep_aspect_ratio: bool  = Form(True),
):
    uploaded = [
        (image_0, mag_weight_0, phase_weight_0),
        (image_1, mag_weight_1, phase_weight_1),
        (image_2, mag_weight_2, phase_weight_2),
        (image_3, mag_weight_3, phase_weight_3),
    ]
 
    images = []
    for upload_file, mw, pw in uploaded:
        if upload_file is not None:
            images.append({
                "bytes":        await upload_file.read(),
                "mag_weight":   mw,
                "phase_weight": pw,
            })
 
    if not images:
        raise HTTPException(status_code=400, detail="No images provided.")
 
    # mix() returns (result_b64, spatial_arr)
    result_b64, spatial_arr = mixer_service.mix(
        images=images,
        component_pair=component_pair,
        region_type=region_type,
        region_size=region_size,
        unify_policy=unify_policy,
        keep_aspect_ratio=keep_aspect_ratio,
    )
 
    # Compute FT components of the mixed result
    arr_norm  = spatial_arr.astype(np.float32) / 255.0
    fft_shift = np.fft.fftshift(np.fft.fft2(arr_norm))
    _, mag_disp, phase_disp, real_disp, imag_disp = prepare_fft_for_display(fft_shift)
 
    def to_b64(a: np.ndarray) -> str:
        uint8 = np.clip(a * 255, 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(uint8).save(buf, format="JPEG", quality=95)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
 
    return JSONResponse({
        "result_image": result_b64,
        "magnitude":    to_b64(mag_disp),
        "phase":        to_b64(phase_disp),
        "real":         to_b64(real_disp),
        "imaginary":    to_b64(imag_disp),
    })
 