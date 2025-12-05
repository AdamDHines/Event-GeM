import os, requests
BACKBONE_URL = ("https://drive.usercontent.google.com/download?id=1kH9GAzNHvEBXIcM69TuCgk9VQ9psQYhe&export=download&authuser=0&confirm=t&uuid=80e16a3c-c7a1-48e0-91a6-ea5be2e9c1f9&at=ALWLOp6BrFTh9Q8kE9jGHzFEID6z%3A1764913935890"
)


def download_backbone_ckpt(dest_path: str, url: str = BACKBONE_URL, chunk_size: int = 1 << 20):
    """
    Download the backbone checkpoint from a direct URL into dest_path.
    Creates parent directories if needed.
    """
    dest_dir = os.path.dirname(dest_path)
    if dest_dir and not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)

    # If it already exists, do nothing
    if os.path.exists(dest_path):
        return

    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)