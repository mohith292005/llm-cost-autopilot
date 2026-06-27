import httpx
import os
from dotenv import load_dotenv

load_dotenv()

r = httpx.get(
    "https://openrouter.ai/api/v1/models",
    headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"}
)

models = r.json()["data"]
free = [m["id"] for m in models if m.get("pricing", {}).get("prompt") == "0"]

print(f"Found {len(free)} free models:\n")
for m in free:
    print(m)