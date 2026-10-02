
import os
from openai import OpenAI

client = OpenAI(
    base_url="http://172.25.16.1:1234/v1",
    api_key="lm-studio"
)

prompt = "Jawab JSON dengan kunci summary, priority, reason, missing_info. Tiket: Pembayaran gagal dan saldo terpotong."

response = client.chat.completions.create(
    model=os.environ["LM_STUDIO_MODEL"],
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0].message.content)
