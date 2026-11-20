import os
import requests
from tqdm import tqdm

BASE = "https://d37ci6vzurychx.cloudfront.net"
files = [f"trip-data/yellow_tripdata_2025-{m:02d}.parquet" for m in range(1, 7)]
files.append("misc/taxi_zone_lookup.csv")

os.makedirs("data", exist_ok=True)

for f in files:
    name = f.split("/")[-1]
    path = os.path.join("data", name)
    if os.path.exists(path):
        print("skip", name)
        continue
    r = requests.get(f"{BASE}/{f}", stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(path, "wb") as out, tqdm(total=total, unit="B", unit_scale=True, desc=name) as bar:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            out.write(chunk)
            bar.update(len(chunk))