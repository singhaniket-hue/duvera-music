"""User-configured visible-browser adapter, or import downloaded Google assets.

No cookie extraction, private endpoints, guessed auth detection, or automatic login.
Selectors must be inspected/configured for the user's current Google interface.
"""
import json
import time
from pathlib import Path
from urllib.parse import urlparse

from ..media import PipelineError
from ..music_policy import instrumental_prompt


def load_browser_config(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    parsed = urlparse(data.get("url", ""))
    if parsed.scheme != "https" or parsed.hostname != "labs.google":
        raise PipelineError("Google browser configuration URL must be on https://labs.google")
    for key in ("prompt", "generate", "download"):
        selector = data.get(key)
        if not isinstance(selector, dict) or set(selector) != {"role", "name"}:
            raise PipelineError(f"Configure {key} with an observed accessible role and name")
    return data


class GoogleFlowProvider:
    def __init__(self, config_path: Path, profile: Path):
        self.config = load_browser_config(config_path)
        self.profile = profile.resolve()
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise PipelineError("Install '.[browser]' and run python -m playwright install chromium") from exc
        self.pw = sync_playwright().start()
        try:
            self.context = self.pw.chromium.launch_persistent_context(
                str(self.profile), headless=False, accept_downloads=True)
            self.page = self.context.pages[0] if self.context.pages else self.context.new_page()
            self.page.goto(self.config["url"], wait_until="domcontentloaded")
            self.locator("prompt").wait_for(state="visible", timeout=20000)
        except Exception as exc:
            self.close()
            raise PipelineError("Google UI/authentication unavailable. Use auth_google_flow.py or import "
                                "downloaded assets. Profile existence does not prove login.") from exc

    def locator(self, name):
        return self.page.get_by_role(**self.config[name], exact=True)

    def generate(self, prompt: str, duration: float, kind: str, destination: Path):
        if kind == "sfx":
            prompt = "Sound effect only, no melody: " + prompt
        else:
            prompt = instrumental_prompt(prompt)
        downloads = self.locator("download")
        previous = downloads.count()
        self.locator("prompt").fill(f"{prompt} Duration approximately {duration:.1f} seconds.")
        self.locator("generate").click()
        # Require a new result. Never download the first stale result from a previous job.
        deadline = time.monotonic() + 300
        while downloads.count() <= previous and time.monotonic() < deadline:
            self.page.wait_for_timeout(1000)
        if downloads.count() <= previous:
            raise PipelineError("No NEW downloadable Google result; UI may have changed. "
                                "No repeated generation was attempted. Import assets instead.")
        with self.page.expect_download(timeout=60000) as item:
            downloads.nth(previous).click()
        item.value.save_as(destination)

    def close(self):
        if hasattr(self, "context"):
            self.context.close()
        if hasattr(self, "pw"):
            self.pw.stop()
