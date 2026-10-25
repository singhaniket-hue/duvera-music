"""Attributed local B-roll, preserving real source pixels and the narrator's soundtrack."""

from __future__ import annotations

import hashlib
import html
from pathlib import Path

from .html_overlay_renderer import _font_css, composite_premultiplied
from .media import PipelineError


def credit_png(label, width, height, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256((label + str((width, height)) + _font_css()).encode()).hexdigest()[:20]
    target = root / (digest + ".png")
    if target.exists():
        return target
    from playwright.sync_api import sync_playwright

    document = f"""<html><head><style>{_font_css()}*{{box-sizing:border-box}}body{{margin:0;background:transparent}}
    .credit{{position:absolute;left:4%;bottom:3%;max-width:90%;padding:15px 24px;background:rgba(10,14,21,.92);
    border-left:4px solid #e9b44c;border-radius:4px;box-shadow:0 7px 25px #0005;color:#f8f5ed;
    font:500 22px/1.5 Mono,Devanagari,sans-serif;white-space:pre-wrap}}</style></head>
    <body><div class="credit">{html.escape(label)}</div></body></html>"""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": height})
        page.set_content(document, wait_until="load")
        page.evaluate("document.fonts.ready")
        page.screenshot(path=str(target), omit_background=True)
        browser.close()
    return target


class MediaCutawayRenderer:
    def __init__(self, events, width, height, root):
        import cv2

        self.width, self.height = width, height
        self.sources = {}
        for event in events:
            if event.edit_type not in {"media_cutaway", "context_note"}:
                continue
            if event.edit_type == "context_note":
                png = credit_png(event.media_credit, width, height, root)
                self.sources[event.event_id] = {
                    "cap": None,
                    "note": True,
                    "credit": cv2.imread(str(png), cv2.IMREAD_UNCHANGED),
                }
                continue
            path = Path(event.media_path)
            if hashlib.sha256(path.read_bytes()).hexdigest() != event.media_sha256:
                raise PipelineError("B-roll source hash changed")
            still = path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            cap = None if still else cv2.VideoCapture(str(path))
            if still:
                frame = cv2.imread(str(path))
                if frame is None:
                    raise PipelineError("Real still image cannot decode")
                png = credit_png(event.media_credit, width, height, root)
                self.sources[event.event_id] = {
                    "cap": None,
                    "fps": 0,
                    "index": 0,
                    "frame": frame,
                    "credit": cv2.imread(str(png), cv2.IMREAD_UNCHANGED),
                }
                continue
            if not cap.isOpened():
                raise PipelineError("Cannot decode real B-roll source")
            fps = cap.get(cv2.CAP_PROP_FPS)
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if (event.media_in_sec + event.end_sec - event.start_sec) * fps > count + 1:
                raise PipelineError(
                    "B-roll hold exceeds real source duration; looping or dummy frames prohibited"
                )
            png = credit_png(event.media_credit, width, height, root)
            self.sources[event.event_id] = {
                "cap": cap,
                "fps": fps,
                "index": -2,
                "frame": None,
                "credit": cv2.imread(str(png), cv2.IMREAD_UNCHANGED),
            }

    def frame(self, event, time, background=None):
        import cv2

        item = self.sources[event.event_id]
        if item.get("note"):
            return composite_premultiplied(background, item["credit"])
        index = int((event.media_in_sec + time - event.start_sec) * item["fps"] + 1e-6)
        if index != item["index"]:
            if index != item["index"] + 1:
                item["cap"].set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = item["cap"].read()
            if not ok:
                raise PipelineError("B-roll ended before its attributed hold")
            item.update(index=index, frame=frame)
        frame = item["frame"]
        h, w = frame.shape[:2]
        scale = min(self.width / w, self.height / h)
        front = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_LANCZOS4)
        scale = max(self.width / w, self.height / h)
        bg = cv2.resize(frame, (round(w * scale), round(h * scale)))
        y, x = (bg.shape[0] - self.height) // 2, (bg.shape[1] - self.width) // 2
        bg = bg[y : y + self.height, x : x + self.width]
        bg = cv2.resize(
            cv2.GaussianBlur(cv2.resize(bg, (self.width // 4, self.height // 4)), (0, 0), 6),
            (self.width, self.height),
        )
        bg = cv2.convertScaleAbs(bg, alpha=0.48)
        y, x = (self.height - front.shape[0]) // 2, (self.width - front.shape[1]) // 2
        bg[y : y + front.shape[0], x : x + front.shape[1]] = front
        return composite_premultiplied(bg, item["credit"])

    def close(self):
        for item in self.sources.values():
            if item["cap"] is not None:
                item["cap"].release()
