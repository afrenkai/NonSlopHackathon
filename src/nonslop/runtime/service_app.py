import json
import re
import time
from io import BytesIO
from pathlib import Path

import modal

from nonslop.runtime.access import configured_access
from nonslop.runtime.web_image import build_web_image

app = modal.App("nonslop-live-checks")
image = build_web_image()
access = configured_access()

with image.imports():
    import httpx
    from PIL import Image

    from nonslop.web.security import environment_settings


@app.function(image=image, secrets=[access], timeout=180)
def public_check(base_url: str) -> dict[str, str]:
    with httpx.Client(base_url=base_url, timeout=120) as client:
        response = client.get("/")
        response.raise_for_status()
        assert response.status_code == 200
        assert "Build your street" in response.text
        assert 'name="api_key"' not in response.text
        assert 'name="password"' not in response.text
        assert re.search(r'name="csrf" value="[^"]+"', response.text)
        assert "httponly" in response.headers["set-cookie"].lower()
        assert client.get("/login").status_code == 303
        assert client.post("/jobs", data={"csrf": "invalid"}).status_code == 403
        assert (
            client.get("/api/jobs/00000000-0000-0000-0000-000000000000").status_code
            == 401
        )
        return {
            "website": base_url,
            "public_http": "200",
            "browser_csrf": "403",
            "api_without_credentials": "401",
        }


@app.function(image=image, secrets=[access], timeout=180)
def submit_live_check(
    base_url: str, reference_image: bytes | None = None
) -> dict[str, str]:
    with httpx.Client(base_url=base_url, timeout=120) as client:
        home = client.get("/")
        home.raise_for_status()
        csrf = re.search(r'name="csrf" value="([^"]+)"', home.text)
        assert csrf is not None
        response = client.post(
            "/jobs",
            data={
                "csrf": csrf.group(1),
                "location": "Hester Street, Lower East Side, New York City",
                "year": "1905",
                "description": "A quiet morning on a narrow street of brick tenements, "
                "shopfronts and a few market carts. Slowly walk forward along the pavement. "
                "Keep the architecture stable and the street lightly occupied.",
                "source_text": "Preview scene inspired by the early twentieth-century Lower East Side.",
                "use_web": "true",
            },
            files=(
                {"reference_image": ("street.png", reference_image, "image/png")}
                if reference_image is not None
                else None
            ),
        )
        response.raise_for_status()
        job_id = re.search(r'id="job-([^"]+)"', response.text)
        assert job_id is not None
        with httpx.Client(base_url=base_url, timeout=120) as visitor:
            assert visitor.get(f"/jobs/{job_id.group(1)}").status_code == 404
            assert visitor.get(f"/jobs/{job_id.group(1)}/reference").status_code == 404
        return {
            "job_id": job_id.group(1),
            "status": "submitted",
            "visitor_access": "404",
        }


@app.function(image=image, secrets=[access], timeout=180)
def live_status(base_url: str, job_id: str) -> dict[str, str]:
    credentials = environment_settings()
    headers = {
        "X-API-Key": credentials.api_key.get_secret_value(),
        "X-Password": credentials.password.get_secret_value(),
    }
    with httpx.Client(base_url=base_url, timeout=120) as client:
        response = client.get(f"/api/jobs/{job_id}", headers=headers)
        response.raise_for_status()
        payload = response.json()
        status = {
            "job_id": payload["job_id"],
            "status": payload["status"],
            "error": payload["error"] or "",
        }
        if payload["status"] == "complete":
            assert client.get(f"/api/jobs/{job_id}/video").status_code == 401
            assert client.get(f"/jobs/{job_id}/video").status_code == 404
            video = client.get(
                f"/api/jobs/{job_id}/video",
                headers={**headers, "Range": "bytes=0-1023"},
            )
            video.raise_for_status()
            assert video.status_code == 206
            assert video.headers["content-type"] == "video/mp4"
            status["video_http"] = str(video.status_code)
            status["video_range"] = video.headers["content-range"]
        return status


@app.function(image=image, timeout=600)
def capsule_check(base_url: str, voice_note: bytes | None = None) -> dict[str, str]:
    picture = BytesIO()
    Image.new("RGB", (16, 16), "red").save(picture, format="PNG")
    with httpx.Client(base_url=base_url, timeout=120) as client:
        response = client.get("/capsules")
        response.raise_for_status()
        token = re.search(r'name="csrf" value="([^"]+)"', response.text)
        assert token is not None
        created = client.post(
            "/capsules",
            data={
                "csrf": token.group(1),
                "title": "Modal capsule smoke test",
                "description": "A test of keeping an original photograph and dated words.",
            },
            files={"anchor": ("test.png", picture.getvalue(), "image/png")},
        )
        created.raise_for_status()
        identifier = re.search(r'data-capsule-id="([^"]+)"', created.text)
        assert identifier is not None
        path = f"/capsules/{identifier.group(1)}"
        original = client.get(f"{path}/anchor")
        original.raise_for_status()
        assert original.content == picture.getvalue()
        added = client.post(
            f"{path}/entries",
            data={
                "csrf": token.group(1),
                "occurred_on": "2006-05-01",
                "text": "My original words remain unchanged.",
            },
        )
        added.raise_for_status()
        assert "01 May 2006" in added.text
        reopened = client.get(path)
        reopened.raise_for_status()
        assert "My original words remain unchanged." in reopened.text
        whole = client.get(f"{path}/entries?view=whole")
        whole.raise_for_status()
        assert "capsule-gallery" in whole.text
        if voice_note is not None:
            recording = client.post(
                f"{path}/entries",
                data={
                    "csrf": token.group(1),
                    "occurred_on": "2007-05-01",
                    "text": "My original recording caption.",
                    "use_ai": "true",
                },
                files={"media": ("question.wav", voice_note, "audio/wav")},
            )
            recording.raise_for_status()
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                transcript = client.get(f"{path}/entries")
                transcript.raise_for_status()
                if "How many cars are there in the picture?" in transcript.text:
                    assert "AI transcription" in transcript.text
                    assert "My original recording caption." in transcript.text
                    break
                time.sleep(2)
            else:
                raise TimeoutError("The live capsule transcription did not finish")
            entry_ids = re.findall(r'data-entry-id="([^"]+)"', transcript.text)
            original_recording = client.get(f"{path}/entries/{entry_ids[-1]}/media")
            original_recording.raise_for_status()
            assert original_recording.content == voice_note
        with httpx.Client(base_url=base_url, timeout=120) as visitor:
            assert visitor.get(path).status_code == 404
            assert visitor.get(f"{path}/anchor").status_code == 404
        return {
            "capsule_id": identifier.group(1),
            "created_and_reopened": "200",
            "original_image": "retained",
            "voice_transcription": (
                "verified" if voice_note is not None else "not_requested"
            ),
            "other_browser": "404",
        }


@app.local_entrypoint()
def check_live(
    base_url: str,
    job_id: str = "",
    image_path: str = "",
    submit: bool = False,
    capsules: bool = False,
    audio_path: str = "",
) -> None:
    if capsules:
        result = capsule_check.remote(
            base_url, Path(audio_path).read_bytes() if audio_path else None
        )
    elif job_id:
        result = live_status.remote(base_url, job_id)
    elif submit:
        result = submit_live_check.remote(
            base_url, Path(image_path).read_bytes() if image_path else None
        )
    else:
        result = public_check.remote(base_url)
    print(json.dumps(result, indent=2))
