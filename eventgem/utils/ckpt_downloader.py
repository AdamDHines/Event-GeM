#!/usr/bin/env python3
from __future__ import annotations

import re
from html import unescape
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import requests


def extract_gdrive_file_id(url: str) -> str:
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)

    if "id" in qs and qs["id"]:
        return qs["id"][0]

    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", parsed.path)
    if m:
        return m.group(1)

    raise ValueError(f"Could not extract Google Drive file ID from URL: {url}")


def parse_hidden_inputs(html: str) -> dict[str, str]:
    """
    Extract hidden input fields from the Drive warning form.
    """
    fields: dict[str, str] = {}

    for match in re.finditer(
        r'<input[^>]+type="hidden"[^>]+name="([^"]+)"[^>]+value="([^"]*)"',
        html,
        flags=re.IGNORECASE,
    ):
        name = unescape(match.group(1))
        value = unescape(match.group(2))
        fields[name] = value

    return fields


def is_html_response(resp: requests.Response) -> bool:
    ctype = resp.headers.get("Content-Type", "").lower()
    return "text/html" in ctype


def get_confirm_form_fields(session: requests.Session, base_url: str, params: dict[str, str], timeout: int) -> dict[str, str]:
    """
    Request the initial URL and, if Google returns a virus-scan warning page,
    parse the hidden form fields needed for the real download.
    """
    resp = session.get(base_url, params=params, stream=True, timeout=timeout)
    resp.raise_for_status()

    if not is_html_response(resp):
        return {}

    html = resp.text
    resp.close()

    if "Google Drive can't scan this file for viruses" not in html and 'id="download-form"' not in html:
        raise RuntimeError("Expected a Google Drive download page, but got unexpected HTML.")

    fields = parse_hidden_inputs(html)
    if "id" not in fields:
        fields["id"] = params["id"]
    if "export" not in fields:
        fields["export"] = "download"

    return fields


def download_google_drive_file(
    chunk_size: int = 1024 * 1024,
    timeout: int = 60,
) -> Path:
    """
    Download a Google Drive file, including files that trigger the
    'can't scan for viruses' interstitial page.
    """
    file_id = extract_gdrive_file_id("https://drive.usercontent.google.com/download?id=1kH9GAzNHvEBXIcM69TuCgk9VQ9psQYhe&export=download&authuser=0")
    output_path = Path('eventgem/ckpt/pr.pt')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_url = "https://drive.usercontent.google.com/download"
    initial_params = {
        "id": file_id,
        "export": "download",
        "authuser": "0",
    }

    with requests.Session() as session:
        form_fields = get_confirm_form_fields(session, base_url, initial_params, timeout)

        # If no HTML warning page was returned, download directly
        final_params = form_fields if form_fields else initial_params

        with session.get(base_url, params=final_params, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()

            if is_html_response(resp):
                preview = resp.text[:1000]
                raise RuntimeError(
                    "Download still returned HTML instead of the file.\n"
                    f"Response starts with:\n{preview}"
                )

            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)

    return output_path