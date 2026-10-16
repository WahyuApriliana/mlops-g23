
import os
import json
import base64
import io
from datetime import date
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError

load_dotenv()

app = FastAPI(
    title="Receipt Finance API",
    version="1.0.0"
)

client = AsyncOpenAI(
    base_url=os.getenv(
        "MODEL_BASE_URL",
        "http://192.168.56.1:1234/v1"
    ),
    api_key=os.getenv("MODEL_API_KEY", "lm-studio"),
    timeout=120.0,
)

MODEL_NAME = os.getenv("MODEL_NAME", "")
MAX_BYTES = int(os.getenv("MAX_FILE_MB", "8")) * 1024 * 1024


class ReceiptItem(BaseModel):
    name: str
    quantity: float
    unit_price: Optional[float] = None
    subtotal: Optional[float] = None


class ReceiptData(BaseModel):
    merchant: Optional[str] = None
    transaction_date: Optional[date] = None
    currency: str = "IDR"
    items: list[ReceiptItem] = Field(default_factory=list)
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    discount: Optional[float] = None
    total: Optional[float] = None
    payment_method: Optional[str] = None


class ReceiptResponse(BaseModel):
    success: bool
    data: ReceiptData
    metadata: dict


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post(
    "/api/receipts/extract",
    response_model=ReceiptResponse
)
async def extract_receipt(file: UploadFile = File(...)):
    if not MODEL_NAME:
        raise HTTPException(
            status_code=500,
            detail="MODEL_NAME belum dikonfigurasi"
        )

    if file.content_type not in {
        "image/jpeg",
        "image/png",
        "image/webp"
    }:
        raise HTTPException(
            status_code=415,
            detail="Format gambar tidak didukung"
        )

    raw = await file.read(MAX_BYTES + 1)

    if not raw or len(raw) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Gambar kosong atau melebihi batas ukuran"
        )

    try:
        image = Image.open(io.BytesIO(raw))
        image.verify()

        image = Image.open(io.BytesIO(raw))

        if image.format not in {"JPEG", "PNG", "WEBP"}:
            raise HTTPException(
                status_code=415,
                detail="Format gambar tidak didukung"
            )

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Berkas bukan gambar yang valid"
        )

    mime = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
    }[image.format]

    encoded = base64.b64encode(raw).decode("ascii")
    image_url = f"data:{mime};base64,{encoded}"

    prompt = """
    Ekstrak data nota keuangan dari gambar.

    Kembalikan satu objek JSON dengan kunci:
    merchant, transaction_date, currency, items, subtotal,
    tax, discount, total, payment_method.

    Setiap item memiliki name, quantity, unit_price, subtotal.
    Gunakan angka untuk nominal, tanggal YYYY-MM-DD,
    dan null untuk data yang tidak diketahui.
    Jangan mengarang informasi.
    Pertahankan total yang tercetak.
    """

    try:
        result = await client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        },
                    ],
                }
            ],
        )

        content = result.choices[0].message.content

        if not content:
            raise ValueError("Output model kosong")

        start = content.find("{")
        end = content.rfind("}")

        if start < 0 or end < start:
            raise ValueError("JSON tidak ditemukan")

        payload = json.loads(content[start:end + 1])
        data = ReceiptData.model_validate(payload)

        return ReceiptResponse(
            success=True,
            data=data,
            metadata={
                "model": MODEL_NAME,
                "processing_status": "success",
            },
        )

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Inferensi gagal atau output model tidak valid"
        )

    finally:
        await file.close()
