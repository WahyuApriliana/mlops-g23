import base64
import json
import os
from pathlib import Path

from openai import OpenAI


# Lokasi gambar nota
IMAGE = Path("data/raw/nota-sample.png")

# Model vision dari LM Studio
MODEL = os.environ["LM_STUDIO_MODEL"]


# Koneksi ke LM Studio
client = OpenAI(
    base_url="http://172.25.16.1:1234/v1",
    api_key="lm-studio"
)


# Baca gambar dan ubah menjadi Base64
image_base64 = base64.b64encode(
    IMAGE.read_bytes()
).decode("utf-8")


# Prompt untuk ekstraksi nota
prompt = """
Baca nota pada gambar dan ekstrak informasi berikut:

- merchant
- tanggal
- item
- subtotal
- pajak
- total

Keluarkan hanya JSON valid dengan struktur berikut:

{
  "merchant": "...",
  "date": "...",
  "items": [],
  "subtotal": 0,
  "tax": 0,
  "total": 0
}

Aturan:
1. Ekstrak informasi sesuai yang terlihat pada nota.
2. Jika pajak tidak terlihat, isi 0.
3. Jangan mengarang informasi.
4. Item harus berisi semua 4 baris barang yang terdapat pada nota.
5. Total harus berupa angka tanpa simbol mata uang.
6. Jangan tambahkan penjelasan di luar JSON.
"""


# Kirim gambar dan prompt ke LM Studio
response = client.chat.completions.create(
    model=MODEL,
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
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                }
            ]
        }
    ]
)


# Ambil jawaban model
content = response.choices[0].message.content

print("HASIL DARI MODEL:")
print(content)


# Bersihkan format Markdown jika model menggunakan ```json
content = content.strip()

if content.startswith("```"):
    content = content.replace("```json", "", 1)
    content = content.replace("```", "", 1)
    content = content.strip()


# Ubah jawaban menjadi JSON
result = json.loads(content)


# Simpan hasil ke reports/receipt.json
output = Path("reports/receipt.json")

output.write_text(
    json.dumps(
        result,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# Tampilkan hasil akhir
print("\nHASIL JSON:")
print(
    json.dumps(
        result,
        indent=2,
        ensure_ascii=False
    )
)

print(f"\nHasil tersimpan di: {output}")
