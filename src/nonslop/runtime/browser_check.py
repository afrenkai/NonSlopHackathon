from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit
from uuid import UUID

import modal

app = modal.App("nonslop-browser-checks")
image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install("playwright", "pydantic")
    .run_commands(
        "uv run --no-project --no-sync -m playwright install --with-deps chromium"
    )
    .uv_pip_install("jinja2")
    .add_local_dir(
        Path(__file__).parents[1] / "web" / "templates",
        "/opt/nonslop-browser/templates",
        copy=True,
    )
    .add_local_dir(
        Path(__file__).parents[1] / "web" / "static",
        "/opt/nonslop-browser/static",
        copy=True,
    )
)
world_volume = modal.Volume.from_name("nonslop-worlds", version=2)

with image.imports():
    from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
    from playwright.sync_api import (
        ConsoleMessage,
        Error,
        Page,
        Request,
        Response,
        Route,
        expect,
        sync_playwright,
    )

    from nonslop.generation.schemas import GenerationResult, JobRecord, SceneRequest
    from nonslop.models.world_schemas import WorldResult
    from nonslop.reconstruction.schemas import ReconstructionRecord


@app.function(image=image, cpu=4, memory=8192, timeout=2700, max_containers=1)
def check(
    base_url: str, reference_images: list[bytes] | None = None
) -> tuple[bytes, bytes, dict[str, str]]:
    console_errors: list[str] = []
    page_errors: list[str] = []
    failed_requests: list[str] = []
    failed_responses: list[str] = []
    response_count = 0
    before = b""
    after = b""
    reference_count = len(reference_images) if reference_images else 0
    metadata = {
        "status": "running",
        "stage": "starting",
        "job_id": "",
        "reference_count": str(reference_count),
    }

    def progress(stage: str) -> None:
        metadata["stage"] = stage
        print(json.dumps({"job_id": metadata["job_id"], "stage": stage}), flush=True)

    def diagnostics() -> dict[str, str]:
        csp_errors = [
            message
            for message in console_errors
            if "content security policy" in message.lower()
            or "violates" in message.lower()
        ]
        return {
            "response_count": str(response_count),
            "page_errors": json.dumps(page_errors),
            "failed_requests": json.dumps(failed_requests),
            "failed_responses": json.dumps(failed_responses),
            "console_errors": json.dumps(console_errors),
            "csp_errors": json.dumps(csp_errors),
            "before_png_bytes": str(len(before)),
            "after_png_bytes": str(len(after)),
        }

    def record_console(message: ConsoleMessage) -> None:
        if (
            message.type == "error"
            and urlsplit(message.location.get("url", "")).path != "/favicon.ico"
        ):
            console_errors.append(message.text)

    def record_page_error(error: Error) -> None:
        page_errors.append(str(error))

    def record_crash(crashed_page: Page) -> None:
        page_errors.append(f"Chromium crashed on {urlsplit(crashed_page.url).path}")

    def record_request_failure(request: Request) -> None:
        path = urlsplit(request.url).path
        failure = request.failure or "Unknown request failure"
        if path != "/favicon.ico" and failure != "net::ERR_ABORTED":
            failed_requests.append(f"{path}: {failure}")

    def record_response(response: Response) -> None:
        nonlocal response_count
        response_count += 1
        path = urlsplit(response.url).path
        if response.status >= 400 and path != "/favicon.ico":
            failed_responses.append(f"{response.status} {path}")

    def is_creation_response(response: Response) -> bool:
        return (
            response.request.method == "POST" and urlsplit(response.url).path == "/jobs"
        )

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            args=[
                "--use-gl=angle",
                "--use-angle=swiftshader",
                "--enable-unsafe-swiftshader",
            ],
        )
        try:
            progress("opening workspace")
            context = browser.new_context(
                base_url=base_url.rstrip("/"),
                viewport={"width": 1024, "height": 900},
                device_scale_factor=1,
            )
            page = context.new_page()
            page.on("console", record_console)
            page.on("pageerror", record_page_error)
            page.on("crash", record_crash)
            page.on("requestfailed", record_request_failure)
            page.on("response", record_response)
            home = page.goto("/", wait_until="domcontentloaded", timeout=120000)
            assert home is not None and home.status == 200
            page.get_by_label("Place or street", exact=True).fill(
                "Hester Street, Lower East Side, New York City"
            )
            page.get_by_label("Year", exact=True).fill("1905")
            page.get_by_label("What would you like to revisit?", exact=True).fill(
                "A quiet morning on a narrow street of brick tenements, shopfronts "
                "and a few market carts. Slowly walk forward along the pavement. "
                "Keep the architecture stable and the street lightly occupied."
            )
            page.get_by_label("Find context on the web").uncheck()
            with TemporaryDirectory() as reference_directory:
                if reference_images:
                    folder_path = Path(reference_directory)
                    for image_number, reference_image in enumerate(
                        reference_images, start=1
                    ):
                        (folder_path / f"{image_number:02d}.png").write_bytes(
                            reference_image
                        )
                    page.locator("summary").filter(
                        has_text="Add multiple reference images"
                    ).click()
                    page.locator("#reference-folder").set_input_files(str(folder_path))
                    noun = "image" if reference_count == 1 else "images"
                    expect(page.locator("#reference-selection")).to_have_text(
                        f"{reference_count} reference {noun} selected."
                    )
                    progress("reference folder selected")
                with page.expect_response(
                    is_creation_response, timeout=120000
                ) as creation:
                    page.get_by_role(
                        "button", name="Create a visit", exact=True
                    ).click()
            assert creation.value.status == 200
            created_article = page.locator("#latest article.visit-card").first
            created_article.wait_for(state="visible", timeout=30000)
            article_id = created_article.get_attribute("id")
            assert article_id is not None and article_id.startswith("job-")
            job_id = UUID(article_id.removeprefix("job-"))
            metadata["job_id"] = str(job_id)
            progress("visit created")
            article = page.locator(f"#job-{job_id}")
            if reference_count > 1:
                expect(article.locator(".reference-gallery img")).to_have_count(
                    reference_count
                )
                metadata["gallery_count"] = str(
                    article.locator(".reference-gallery img").count()
                )
            build_button = article.get_by_role(
                "button", name="Build a 3D space", exact=True
            )
            progress("waiting for video")
            build_button.wait_for(state="visible", timeout=1200000)
            progress("video ready")
            build_button.click()
            progress("reconstruction requested")
            world_link = article.get_by_role(
                "link", name="Step into this space", exact=True
            )
            world_link.wait_for(state="visible", timeout=300000)
            progress("3D space ready")
            world_link.click()
            page.wait_for_url(f"**/jobs/{job_id}/world", timeout=30000)
            page.locator("#world-status").filter(
                has_text="Your 3D visit is ready"
            ).wait_for(state="visible", timeout=180000)
            canvas = page.locator("#world-canvas")
            assert canvas.evaluate("canvas => Boolean(canvas.getContext('webgl2'))")
            metadata["world_url"] = page.url
            metadata["world_status"] = page.locator("#world-status").inner_text()
            metadata["camera_count"] = page.locator("#world-camera-count").inner_text()
            metadata["webgl2"] = "true"
            progress("viewer ready")
            canvas.click()
            progress("capturing before screenshot")
            before = canvas.screenshot(
                type="png", animations="disabled", timeout=180000
            )
            progress("before screenshot captured")
            page.keyboard.press("w", delay=1500)
            progress("walked forward")
            progress("capturing after screenshot")
            after = canvas.screenshot(type="png", animations="disabled", timeout=180000)
            progress("after screenshot captured")

            other_context = browser.new_context(base_url=base_url.rstrip("/"))
            try:
                private_video = other_context.request.get(
                    f"/jobs/{job_id}/video",
                    headers={"Range": "bytes=0-0"},
                    timeout=30000,
                )
                video_status = private_video.status
                private_video.dispose()
                private_world = other_context.request.get(
                    f"/jobs/{job_id}/world/gaussians.ply",
                    headers={"Range": "bytes=0-0"},
                    timeout=30000,
                )
                world_status = private_world.status
                private_world.dispose()
            finally:
                other_context.close()

            metadata["other_workspace_video_status"] = str(video_status)
            metadata["other_workspace_world_status"] = str(world_status)
            metadata.update(diagnostics())
            assert not page_errors, metadata
            assert not failed_requests, metadata
            assert not failed_responses, metadata
            assert metadata["csp_errors"] == "[]", metadata
            assert video_status == 404 and world_status == 404, metadata
            metadata["status"] = "passed"
            progress("checks passed")
            return before, after, metadata
        except (Error, AssertionError) as error:
            metadata.update(diagnostics())
            metadata["status"] = "failed"
            metadata["error"] = str(error)
            print(
                json.dumps(
                    {
                        "job_id": metadata["job_id"],
                        "stage": metadata["stage"],
                        "status": "failed",
                    }
                ),
                flush=True,
            )
            return before, after, metadata
        finally:
            browser.close()


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=1200,
    max_containers=1,
    volumes={"/worlds": world_volume},
)
def check_navigation(base_url: str, job_id: str) -> tuple[bytes, bytes, dict[str, str]]:
    identifier = UUID(job_id)
    world_volume.reload()
    artifact_root = Path("/worlds") / str(identifier)
    before = b""
    after = b""
    console_errors: list[str] = []
    page_errors: list[str] = []
    failed_requests: list[str] = []
    failed_responses: list[str] = []
    metadata = {
        "mode": "navigation-fixture",
        "job_id": str(identifier),
        "status": "running",
        "stage": "starting",
    }

    def progress(stage: str) -> None:
        metadata["stage"] = stage
        print(json.dumps({"job_id": metadata["job_id"], "stage": stage}), flush=True)

    def diagnostics() -> dict[str, str]:
        return {
            "console_errors": json.dumps(console_errors),
            "page_errors": json.dumps(page_errors),
            "failed_requests": json.dumps(failed_requests),
            "failed_responses": json.dumps(failed_responses),
            "csp_errors": json.dumps(
                [
                    message
                    for message in console_errors
                    if "content security policy" in message.lower()
                    or "violates" in message.lower()
                ]
            ),
            "before_png_bytes": str(len(before)),
            "after_png_bytes": str(len(after)),
        }

    def record_console(message: ConsoleMessage) -> None:
        if (
            message.type == "error"
            and urlsplit(message.location.get("url", "")).path != "/favicon.ico"
        ):
            console_errors.append(message.text)

    def record_page_error(error: Error) -> None:
        page_errors.append(str(error))

    def record_request_failure(request: Request) -> None:
        path = urlsplit(request.url).path
        failure = request.failure or "Unknown request failure"
        if path != "/favicon.ico" and failure != "net::ERR_ABORTED":
            failed_requests.append(f"{path}: {failure}")

    def record_response(response: Response) -> None:
        path = urlsplit(response.url).path
        if response.status >= 400 and path != "/favicon.ico":
            failed_responses.append(f"{response.status} {path}")

    fixture = JobRecord(
        job_id=str(identifier),
        owner_id="browser-check-fixture",
        created_at=datetime.now(UTC),
        status="complete",
        scene=SceneRequest(
            location="Hester Street, Lower East Side, New York City",
            year=1905,
            description="A quiet morning on a narrow street of brick tenements, "
            "shopfronts and a few market carts. Slowly walk forward along the pavement. "
            "Keep the architecture stable and the street lightly occupied.",
            use_web=False,
        ),
        render_prompt="Saved generated street for isolated viewer navigation checks.",
        result=GenerationResult(
            video_path=f"/outputs/{identifier}.mp4", width=768, height=448, frames=124
        ),
        world=ReconstructionRecord(
            status="complete",
            result=WorldResult(
                job_id=identifier,
                ply_path=str(artifact_root / "gaussians.ply"),
                camera_path=str(artifact_root / "camera_params.json"),
            ),
        ),
    )
    templates = Environment(
        loader=FileSystemLoader("/opt/nonslop-browser/templates"),
        undefined=StrictUndefined,
        autoescape=select_autoescape(("html",)),
    )
    world_html = templates.get_template("world.html").render(
        job=fixture, linked_memories=[]
    )

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            args=[
                "--use-gl=angle",
                "--use-angle=swiftshader",
                "--enable-unsafe-swiftshader",
            ],
        )
        try:
            context = browser.new_context(
                base_url=base_url.rstrip("/"),
                viewport={"width": 1024, "height": 900},
                device_scale_factor=1,
            )
            progress("reading deployed CSP")
            homepage = context.request.get("/", timeout=120000)
            assert homepage.status == 200
            policy = homepage.headers.get("content-security-policy", "")
            assert policy, "The deployed homepage must provide its CSP"
            homepage.dispose()
            page = context.new_page()
            page.on("console", record_console)
            page.on("pageerror", record_page_error)
            page.on("requestfailed", record_request_failure)
            page.on("response", record_response)

            def serve_world(route: Route) -> None:
                route.fulfill(
                    status=200,
                    body=world_html,
                    headers={
                        "Content-Type": "text/html; charset=utf-8",
                        "Content-Security-Policy": policy,
                        "Cache-Control": "no-store",
                    },
                )

            def serve_cameras(route: Route) -> None:
                route.fulfill(
                    status=200,
                    path=str(artifact_root / "camera_params.json"),
                    content_type="application/json",
                )

            def serve_splats(route: Route) -> None:
                route.fulfill(
                    status=200,
                    path=str(artifact_root / "gaussians.ply"),
                    content_type="application/octet-stream",
                )

            world_path = f"/jobs/{identifier}/world"
            page.route(f"**{world_path}", serve_world)
            page.route(f"**{world_path}/cameras", serve_cameras)
            page.route(f"**{world_path}/gaussians.ply", serve_splats)
            progress("loading persisted world fixture")
            page.goto(world_path, wait_until="domcontentloaded", timeout=120000)
            status = page.locator("#world-status")
            expect(status).to_contain_text("Your 3D visit is ready", timeout=180000)
            canvas = page.locator("#world-canvas")
            assert canvas.evaluate("canvas => Boolean(canvas.getContext('webgl2'))")
            metadata["world_url"] = page.url
            metadata["camera_count"] = page.locator("#world-camera-count").inner_text()
            metadata["webgl2"] = "true"
            canvas.focus()
            progress("capturing before screenshot")
            before = canvas.screenshot(
                type="png", animations="disabled", timeout=180000
            )
            progress("holding W until movement updates")
            page.keyboard.down("w")
            try:
                expect(status).to_have_text(
                    "Exploring the reconstructed space. Reset view to return to the photograph.",
                    timeout=180000,
                )
            finally:
                page.keyboard.up("w")
            metadata["movement_status"] = status.inner_text()
            progress("capturing after screenshot")
            after = canvas.screenshot(type="png", animations="disabled", timeout=180000)
            progress("resetting view")
            page.get_by_role("button", name="Reset view", exact=True).click()
            expect(status).to_contain_text("Your 3D visit is ready", timeout=180000)
            metadata["reset_status"] = status.inner_text()
            metadata.update(diagnostics())
            assert not page_errors, metadata
            assert not failed_requests, metadata
            assert not failed_responses, metadata
            assert metadata["csp_errors"] == "[]", metadata
            metadata["status"] = "passed"
            progress("navigation checks passed")
            return before, after, metadata
        except (Error, AssertionError) as error:
            metadata.update(diagnostics())
            metadata["status"] = "failed"
            metadata["error"] = str(error)
            progress(f"failed: {metadata['stage']}")
            return before, after, metadata
        finally:
            browser.close()


@app.local_entrypoint()
def main(
    base_url: str,
    image_path: str = "",
    last_image_path: str = "",
    navigation_job_id: str = "",
) -> None:
    if navigation_job_id:
        if image_path or last_image_path:
            raise ValueError(
                "Navigation checks reuse saved geometry and accept no images"
            )
        before, after, metadata = check_navigation.remote(base_url, navigation_job_id)
        before_path = Path("/tmp/nonslop-navigation-before.png")
        after_path = Path("/tmp/nonslop-navigation-after.png")
    else:
        if last_image_path and not image_path:
            raise ValueError("--last-image-path requires --image-path")
        reference_images = [Path(image_path).read_bytes()] if image_path else None
        if last_image_path and reference_images is not None:
            reference_images.append(Path(last_image_path).read_bytes())
        before, after, metadata = check.remote(base_url, reference_images)
        before_path = Path("/tmp/nonslop-world-before.png")
        after_path = Path("/tmp/nonslop-world-after.png")
    if before:
        before_path.write_bytes(before)
    if after:
        after_path.write_bytes(after)
    metadata["before_path"] = str(before_path) if before else ""
    metadata["after_path"] = str(after_path) if after else ""
    print(json.dumps(metadata, indent=2))
    if metadata["status"] == "failed":
        raise SystemExit(1)
