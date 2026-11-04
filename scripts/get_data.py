"""Download MovieLens-1M from GroupLens into data/raw/.

The dataset is not redistributed in this repo (GroupLens terms). Usage: python scripts/get_data.py
"""
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
out = Path(__file__).resolve().parent.parent / "data" / "raw"
out.mkdir(parents=True, exist_ok=True)
print(f"downloading {URL} ...")
z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL).read()))
for name in ("ratings.dat", "movies.dat", "users.dat"):
    (out / name).write_bytes(z.read(f"ml-1m/{name}"))
    print("wrote", out / name)
