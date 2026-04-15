import io
import json
import base64
import numpy as np
from PIL import Image
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from Objects.FFT_Applicable_Image      import FFT_Applicable_Image
from Objects.FourImagesMixer import FourImagesMixer
from Objects.ImagePropertiesEmphasizer import ImagePropertiesEmphasizer
from Services.FrequencyTransformService import arr_to_bytes

# py -m uvicorn endpoints:app --reload

app = FastAPI()

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Long-lived emphasizer instance — replaced only when a new image is uploaded
# ─────────────────────────────────────────────────────────────────────────────

_active_emphasizer: Optional[ImagePropertiesEmphasizer] = None


def _get_emphasizer() -> ImagePropertiesEmphasizer:
    """Return the active emphasizer or raise a clean HTTP 400."""
    if _active_emphasizer is None:
        raise HTTPException(
            status_code=400,
            detail="No image loaded. POST to /api/properties/upload first.",
        )
    return _active_emphasizer


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

async def parse_image_input(
    image:         Optional[UploadFile],
    complex_input: Optional[str],
) -> np.ndarray:
    """
    Accepts either a raw image file upload or a JSON-encoded complex array.
    Returns a float32 (real) or complex64 numpy array.
    """
    if complex_input is not None:
        try:
            data      = json.loads(complex_input)
            shape     = tuple(data["shape"])
            real_part = np.array(data["real"], dtype=np.float32).reshape(shape)
            if data.get("imag") is not None:
                imag_part = np.array(data["imag"], dtype=np.float32).reshape(shape)
                return (real_part + 1j * imag_part).astype(np.complex64)
            return real_part
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Malformed complex_input JSON: {exc}",
            )

    if image is not None:
        img_data  = await image.read()
        pil_image = Image.open(io.BytesIO(img_data)).convert("L")
        return np.array(pil_image, dtype=np.float32) / 255.0

    raise HTTPException(
        status_code=400,
        detail="Provide either 'image' (file upload) or 'complex_input' (JSON).",
    )


def _build_part(boundary: str, name: str, content: bytes, ct: str = "image/jpeg") -> bytes:
    hdr = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{name}"\r\n'
        f"Content-Type: {ct}\r\n\r\n"
    ).encode()
    return hdr + content + b"\r\n"


def _fft_comps_to_parts(boundary: str, fft_comps: dict) -> bytes:
    """Serialise all 8 FFT display images into multipart body bytes."""
    body = b""
    body += _build_part(boundary, "magnitude",           arr_to_bytes(fft_comps["shifted_mag"]))
    body += _build_part(boundary, "phase",               arr_to_bytes(fft_comps["shifted_phase"]))
    body += _build_part(boundary, "real",                arr_to_bytes(fft_comps["shifted_real"]))
    body += _build_part(boundary, "imaginary",           arr_to_bytes(fft_comps["shifted_imag"]))
    body += _build_part(boundary, "unshifted_magnitude", arr_to_bytes(fft_comps["unshifted_mag"]))
    body += _build_part(boundary, "unshifted_phase",     arr_to_bytes(fft_comps["unshifted_phase"]))
    body += _build_part(boundary, "unshifted_real",      arr_to_bytes(fft_comps["unshifted_real"]))
    body += _build_part(boundary, "unshifted_imaginary", arr_to_bytes(fft_comps["unshifted_imag"]))
    return body


def _apply_operation(
    img: ImagePropertiesEmphasizer,
    action:         str,
    shift_x:        int,
    shift_y:        int,
    cyclic:         bool,
    flip:           bool,
    stretch_x:      float,
    stretch_y:      float,
    angle:          float,
    mirror_axis:    str,
    duplicate_mode: bool,
    amplitude:      float,
    freq_u:         float,
    freq_v:         float,
    window_type:    str,
    window_width:   int,
    window_height:  int,
    center_x:       int,
    center_y:       int,
    sigma_x:        float,
    sigma_y:        float,
    symmetry_type:  str,
    axis:           str,
) -> ImagePropertiesEmphasizer:
    """
    Dispatch table — applies the requested operation to an emphasizer and
    returns a new emphasizer wrapping the result.  Shared by both endpoints.
    """
    if action == "shift":
        return img.shift(shift_x, shift_y, cyclic=cyclic, flip=flip)
    elif action == "stretch":
        return img.stretch(stretch_x, stretch_y)
    elif action == "rotate":
        return img.rotate(angle)
    elif action == "mirror":
        return img.mirror(mirror_axis, duplicate_mode)
    elif action == "even":
        return img.make_even_or_odd("even")
    elif action == "odd":
        return img.make_even_or_odd("odd")
    elif action == "complex_exp":
        return img.multiply_by_complex_exponential(amplitude, freq_u, freq_v)
    elif action == "window":
        return img.multiply_by_window(
            window_width, window_height, center_x, center_y,
            window_type=window_type, sigma_x=sigma_x, sigma_y=sigma_y,
        )
    elif action == "differentiate":
        return img.differentiate(axis)
    elif action == "integrate":
        return img.integrate(axis)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")


# ─────────────────────────────────────────────────────────────────────────────
# Shared form parameters (DRY — used by both operation endpoints)
# ─────────────────────────────────────────────────────────────────────────────

OPERATION_FORM_DEFAULTS = dict(
    action        = Form(...),
    shift_x       = Form(0),
    shift_y       = Form(0),
    cyclic        = Form(True),
    flip          = Form(False),
    stretch_x     = Form(1.0),
    stretch_y     = Form(1.0),
    angle         = Form(0.0),
    mirror_axis   = Form("horizontal"),
    duplicate_mode= Form(False),
    amplitude     = Form(1.0),
    freq_u        = Form(0.0),
    freq_v        = Form(0.0),
    window_type   = Form("gaussian"),
    window_width  = Form(256),
    window_height = Form(256),
    center_x      = Form(0),
    center_y      = Form(0),
    sigma_x       = Form(30.0),
    sigma_y       = Form(30.0),
    symmetry_type = Form("even"),
    axis          = Form("x"),
)


# ─────────────────────────────────────────────────────────────────────────────
# /api/properties/upload
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/properties/upload")
async def UploadPropertiesImageEndpoint(
    image:         Optional[UploadFile] = File(None),
    complex_input: Optional[str]        = Form(None),
):
    """
    Loads a new image into the long-lived ImagePropertiesEmphasizer.
    Must be called before operate_then_fft or fft_then_operate.
    Returns the original image + its FFT components as base64 JPEGs.
    """
    global _active_emphasizer

    raw                = await parse_image_input(image, complex_input)
    _active_emphasizer = ImagePropertiesEmphasizer(array=raw)

    components = _active_emphasizer.get_fft_components()

    def to_b64(a: np.ndarray) -> str:
        uint8 = np.clip(a * 255, 0, 255).astype(np.uint8)
        buf   = io.BytesIO()
        Image.fromarray(uint8).save(buf, format="JPEG", quality=95)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    return JSONResponse({
        "original":  _active_emphasizer.to_base64(),
        "magnitude": to_b64(components["shifted_mag"]),
        "phase":     to_b64(components["shifted_phase"]),
        "real":      to_b64(components["shifted_real"]),
        "imaginary": to_b64(components["shifted_imag"]),
    })


# ─────────────────────────────────────────────────────────────────────────────
# /operate_then_fft
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/operate_then_fft")
async def OperateThenFftEndpoint(
    action:         str   = Form(...),
    shift_x:        int   = Form(0),
    shift_y:        int   = Form(0),
    cyclic:         bool  = Form(True),
    flip:           bool  = Form(False),
    stretch_x:      float = Form(1.0),
    stretch_y:      float = Form(1.0),
    angle:          float = Form(0.0),
    mirror_axis:    str   = Form("horizontal"),
    duplicate_mode: bool  = Form(False),
    amplitude:      float = Form(1.0),
    freq_u:         float = Form(0.0),
    freq_v:         float = Form(0.0),
    window_type:    str   = Form("gaussian"),
    window_width:   int   = Form(256),
    window_height:  int   = Form(256),
    center_x:       int   = Form(0),
    center_y:       int   = Form(0),
    sigma_x:        float = Form(30.0),
    sigma_y:        float = Form(30.0),
    symmetry_type:  str   = Form("even"),
    axis:           str   = Form("x"),
):
    # ── 1. Pull the long-lived instance (no image bytes needed) ──────────
    img = _get_emphasizer()

    # ── 2. Apply the spatial operation on the stored original array ───────
    result_img = _apply_operation(
        img, action,
        shift_x, shift_y, cyclic, flip,
        stretch_x, stretch_y,
        angle,
        mirror_axis, duplicate_mode,
        amplitude, freq_u, freq_v,
        window_type, window_width, window_height, center_x, center_y,
        sigma_x, sigma_y,
        symmetry_type, axis,
    )

    operated = result_img.array

    # ── 3. Build the spatial display image ────────────────────────────────
    # complex_exp is the only action that produces a meaningful complex
    # spatial result that needs all 4 spatial components shown separately.
    spatial_is_complex = result_img.is_complex() and action == "complex_exp"

    if spatial_is_complex:
        sp_comps      = result_img.get_complex_spatial_components()
        spatial_bytes = arr_to_bytes(sp_comps["magnitude"])
    else:
        display = np.abs(operated) if result_img.is_complex() else operated
        lo, hi  = display.min(), display.max()
        norm    = (display - lo) / (hi - lo) if hi > lo else np.zeros_like(display)
        spatial_bytes = arr_to_bytes(norm.astype(np.float32))

    # ── 4. Compute FFT display components of the operated result ──────────
    fft_comps = result_img.get_fft_components()

    # ── 5. Assemble multipart response ────────────────────────────────────
    boundary = "operate_fft_boundary"
    body     = b""

    body += _build_part(boundary, "spatial", spatial_bytes)

    # Only include the 4 spatial sub-components when the result is complex
    if spatial_is_complex:
        body += _build_part(boundary, "spatial_magnitude", arr_to_bytes(sp_comps["magnitude"]))
        body += _build_part(boundary, "spatial_phase",     arr_to_bytes(sp_comps["phase"]))
        body += _build_part(boundary, "spatial_real",      arr_to_bytes(sp_comps["real"]))
        body += _build_part(boundary, "spatial_imaginary", arr_to_bytes(sp_comps["imaginary"]))

    body += _fft_comps_to_parts(boundary, fft_comps)
    body += f"--{boundary}--\r\n".encode()

    return StreamingResponse(
        io.BytesIO(body),
        media_type=f"multipart/form-data; boundary={boundary}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# /fft_then_operate
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/fft_then_operate")
async def FftThenOperateEndpoint(
    action:         str   = Form(...),
    shift_x:        int   = Form(0),
    shift_y:        int   = Form(0),
    cyclic:         bool  = Form(True),
    flip:           bool  = Form(False),
    stretch_x:      float = Form(1.0),
    stretch_y:      float = Form(1.0),
    angle:          float = Form(0.0),
    mirror_axis:    str   = Form("horizontal"),
    duplicate_mode: bool  = Form(False),
    amplitude:      float = Form(1.0),
    freq_u:         float = Form(0.0),
    freq_v:         float = Form(0.0),
    window_type:    str   = Form("gaussian"),
    window_width:   int   = Form(256),
    window_height:  int   = Form(256),
    center_x:       int   = Form(0),
    center_y:       int   = Form(0),
    sigma_x:        float = Form(30.0),
    sigma_y:        float = Form(30.0),
    symmetry_type:  str   = Form("even"),
    axis:           str   = Form("x"),
    scenario_type:  str   = Form("B"),
    n:              int   = Form(1),
):
    # ── 1. Pull the long-lived instance ───────────────────────────────────
    img = _get_emphasizer()

    # ── 2. Get the raw FFT and SHIFT it so DC is in the center ────────────
    #    All operations (shift, window, mirror, etc.) are only meaningful
    #    when the DC component is centered. Without this, a "shift" in the
    #    frequency domain operates on the wrong quadrant layout and the
    #    IFFT produces garbage.
    raw_fft        = img.get_raw_fft()
    fft_shifted    = np.fft.fftshift(raw_fft)
    fft_img        = ImagePropertiesEmphasizer(array=fft_shifted)

    # ── 3. Apply operation on the DC-centered FFT ─────────────────────────
    result_img = _apply_operation(
        fft_img, action,
        shift_x, shift_y, cyclic, flip,
        stretch_x, stretch_y,
        angle,
        mirror_axis, duplicate_mode,
        amplitude, freq_u, freq_v,
        window_type, window_width, window_height, center_x, center_y,
        sigma_x, sigma_y,
        symmetry_type, axis,
    )

    operated = result_img.array   # still complex, still DC-centered

    # ── 4. IFFT ───────────────────────────────────────────────────────────
    #    The operated array is DC-centered so we MUST ifftshift before
    #    ifft2, otherwise the spatial result is completely wrong.
    spatial_complex = np.fft.ifft2(np.fft.ifftshift(operated))

    # ── 5. Spatial display ────────────────────────────────────────────────
    spatial_is_complex = action == "shift"

    if spatial_is_complex:
        spatial_img   = ImagePropertiesEmphasizer(array=spatial_complex)
        sp_comps      = spatial_img.get_complex_spatial_components()
        spatial_bytes = arr_to_bytes(sp_comps["magnitude"])
    else:
        spatial_real  = np.real(spatial_complex)
        lo, hi        = spatial_real.min(), spatial_real.max()
        norm          = (spatial_real - lo) / (hi - lo) if hi > lo else np.zeros_like(spatial_real)
        spatial_bytes = arr_to_bytes(norm.astype(np.float32))

    # ── 6. FFT display of the operated frequency data ─────────────────────
    #    The result_img still holds the DC-centered operated FFT.
    #    get_fft_components() will call fft2 on it which is wrong — we
    #    want to DISPLAY the operated FFT directly, not FFT it again.
    #    So we call _prepare_fft_for_display directly on the operated array.
    (
        _,
        shifted_mag,   shifted_phase,   shifted_real,   shifted_imag,
        unshifted_mag, unshifted_phase, unshifted_real, unshifted_imag,
    ) = FFT_Applicable_Image._prepare_fft_for_display(np.fft.ifftshift(operated))

    fft_comps = {
        "shifted_mag":      shifted_mag,
        "shifted_phase":    shifted_phase,
        "shifted_real":     shifted_real,
        "shifted_imag":     shifted_imag,
        "unshifted_mag":    unshifted_mag,
        "unshifted_phase":  unshifted_phase,
        "unshifted_real":   unshifted_real,
        "unshifted_imag":   unshifted_imag,
    }

    # ── 7. Assemble multipart response ────────────────────────────────────
    boundary = "fft_operate_boundary"
    body     = b""

    body += _build_part(boundary, "spatial", spatial_bytes)

    if spatial_is_complex:
        body += _build_part(boundary, "spatial_magnitude", arr_to_bytes(sp_comps["magnitude"]))
        body += _build_part(boundary, "spatial_phase",     arr_to_bytes(sp_comps["phase"]))
        body += _build_part(boundary, "spatial_real",      arr_to_bytes(sp_comps["real"]))
        body += _build_part(boundary, "spatial_imaginary", arr_to_bytes(sp_comps["imaginary"]))

    body += _fft_comps_to_parts(boundary, fft_comps)
    body += f"--{boundary}--\r\n".encode()

    return StreamingResponse(
        io.BytesIO(body),
        media_type=f"multipart/form-data; boundary={boundary}",
    )


## Mixer Code
# ─────────────────────────────────────────────────────────────────────────────
# Long-lived mixer instance — slots replaced only when new images are uploaded
# ─────────────────────────────────────────────────────────────────────────────

mixer = FourImagesMixer()


def _get_mixer_slot(slot: int) -> None:
    """Validate slot index."""
    if not 0 <= slot <= 3:
        raise HTTPException(
            status_code=400,
            detail=f"Slot index must be 0–3, got {slot}.",
        )


# ─────────────────────────────────────────────────────────────────────────────
# /api/mixer/upload
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/mixer/upload")
async def UploadMixerImageEndpoint(
    slot:         int   = Form(...),
    mag_weight:   float = Form(1.0),
    phase_weight: float = Form(1.0),
    image:        UploadFile = File(...),
):
    """
    Load or replace one of the four mixer image slots.
    The FFT is computed immediately and cached inside the slot.
    Returns the uploaded image + its FFT components as base64 JPEGs
    so the UI can display them right after upload.
    """
    _get_mixer_slot(slot)

    image_bytes = await image.read()

    mixer.set_image(
        index        = slot,
        image_bytes  = image_bytes,
        mag_weight   = mag_weight,
        phase_weight = phase_weight,
    )

    # Build a temporary FFT_Applicable_Image just to return display components
    # to the UI — the real one lives inside the mixer slot.
    img_obj    = FFT_Applicable_Image(image_bytes=image_bytes)
    components = img_obj.get_fft_components()

    def to_b64(a: np.ndarray) -> str:
        uint8 = np.clip(a * 255, 0, 255).astype(np.uint8)
        buf   = io.BytesIO()
        Image.fromarray(uint8).save(buf, format="JPEG", quality=95)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    return JSONResponse({
        "status":    "ok",
        "slot":      slot,
        "original":  img_obj.to_base64(),
        "magnitude": to_b64(components["shifted_mag"]),
        "phase":     to_b64(components["shifted_phase"]),
        "real":      to_b64(components["shifted_real"]),
        "imaginary": to_b64(components["shifted_imag"]),
    })


# ─────────────────────────────────────────────────────────────────────────────
# /api/mixer/run
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/mixer/run")
async def RunMixEndpoint(
    component_pair:    str   = Form("mag-phase"),
    region_type:       str   = Form("inner"),
    region_size:       float = Form(40.0),
    unify_policy:      str   = Form("smallest"),
    keep_aspect_ratio: bool  = Form(True),
    # Per-slot weight overrides — only applied to slots that are already loaded
    mag_weight_0:   float = Form(1.0),
    mag_weight_1:   float = Form(1.0),
    mag_weight_2:   float = Form(1.0),
    mag_weight_3:   float = Form(1.0),
    phase_weight_0: float = Form(1.0),
    phase_weight_1: float = Form(1.0),
    phase_weight_2: float = Form(1.0),
    phase_weight_3: float = Form(1.0),
):
    """
    Run the weighted FT mix across all currently loaded mixer slots.
    No image upload needed — uses whatever is already in the slots.
    Weight overrides are applied before mixing so the UI can tweak
    weights without re-uploading images.
    """
    # ── 1. Apply weight overrides to all populated slots ──────────────────
    weight_overrides = [
        (0, mag_weight_0, phase_weight_0),
        (1, mag_weight_1, phase_weight_1),
        (2, mag_weight_2, phase_weight_2),
        (3, mag_weight_3, phase_weight_3),
    ]
    for slot_idx, mw, pw in weight_overrides:
        if mixer._slots[slot_idx] is not None:
            mixer.update_weights(slot_idx, mw, pw)

    # ── 2. Run the mix ────────────────────────────────────────────────────
    try:
        result = mixer.mix(
            component_pair    = component_pair,
            region_type       = region_type,
            region_size       = region_size,
            unify_policy      = unify_policy,
            keep_aspect_ratio = keep_aspect_ratio,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # ── 3. Return result image + FT display components ────────────────────
    # spatial_arr is a uint8 numpy array — not JSON-serialisable, drop it
    spatial_arr = result.pop("spatial_arr")

    return JSONResponse(result)


# ─────────────────────────────────────────────────────────────────────────────
# /api/mixer/clear
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/mixer/clear")
async def ClearMixerSlotEndpoint(
    slot: int = Form(...),
):
    """Clear a single mixer slot."""
    _get_mixer_slot(slot)
    mixer.clear_slot(slot)
    return JSONResponse({"status": "ok", "slot": slot, "cleared": True})