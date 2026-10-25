"""Seekable Chromium/WebAnimations documentary graphics, rendered to transparent PNGs."""
from __future__ import annotations

import base64
import hashlib
import html
import json
import math
import re
import shutil
import mimetypes
from pathlib import Path

import numpy as np

from .media import PipelineError

VERSION = "html-cinema-v2-1"
FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"


def _font_css():
    rules = []
    for family, pattern in (("Editorial", "PlusJakartaSans*.ttf"), ("Mono", "JetBrainsMono*.ttf"),
                            ("Devanagari", "NotoSansDevanagari*.ttf")):
        files = sorted(FONT_DIR.glob(pattern))
        if files:
            data = base64.b64encode(files[0].read_bytes()).decode()
            rules.append(f"@font-face{{font-family:{family};src:url(data:font/ttf;base64,{data});font-weight:100 900}}")
    if len(rules) < 2:
        raise PipelineError("Install the licensed editorial/mono fonts in assets/fonts before rendering")
    return "\n".join(rules)


def _normal(text):
    return re.sub(r"[^\w\u0900-\u097f]", "", text.casefold()).replace("।", "").replace("॥", "")


def kinetic_word_times(event, words):
    """Match displayed words to actual ASR tokens; report any unaligned fallback."""
    if event.edit_type == 'slogan_cards':
        times=[max(0,t-event.start_sec) for line in event.quote_lines for t in line.word_times]
        return times,{'timing_source':'asr_words','matched_words':len(times),'displayed_words':len(times)}
    if event.edit_type != "black_chapter_card":
        return [], {"timing_source": "not_applicable_static_editorial_text"}
    candidates = [w for w in words if event.start_sec - .25 <= w["start"] < event.end_sec]
    times, index, matched = [], 0, 0
    tokens = (event.headline_text or "").split()
    for n, token in enumerate(tokens):
        found = next((i for i in range(index, len(candidates))
                      if _normal(candidates[i]["word"]) == _normal(token)), None)
        if found is not None:
            times.append(max(0, candidates[found]["start"] - event.start_sec))
            index, matched = found + 1, matched + 1
        else:
            times.append(.22 + n * .12)
    return times, {"matched_words": matched, "displayed_words": len(tokens),
                   "timing_source": "asr_words" if matched == len(tokens) else "mixed_asr_and_editorial_fallback"}


def build_html(event, width, height, word_times=None):
    text = event.headline_text or ""
    safe = html.escape(text)
    if event.edit_type=='slogan_cards':
        safe=''.join('<div class="quote"><div class="quote-number">'+f'0{i+1} // '+
            ('RESPONSE' if event.quote_layout=='side_by_side' else 'SLOGAN')+'</div><div>'+''.join(
                '<span class="word">'+html.escape(w)+'</span> ' for w in line.text.split())+'</div></div>'
            for i,line in enumerate(event.quote_lines))
    elif event.edit_type == "black_chapter_card":
        safe = " ".join(f'<span class="word">{html.escape(w)}</span>' for w in text.split())
    else:
        phrases = getattr(event, "highlight_phrases", []) or re.findall(r"(?:₹|\$)?\d[\d,.]*(?:%|\s+(?:Crore|crore|lakh))?", text)
        for phrase in sorted(set(phrases), key=len, reverse=True):
            escaped = html.escape(phrase)
            safe = safe.replace(escaped, f'<mark><span class="sweep"></span><span class="ink">{escaped}</span></mark>', 1)
    data = {"kind": event.edit_type, "duration": (event.end_sec - event.start_sec) * 1000,
            "times": [t * 1000 for t in (word_times or [])]}
    metadata = getattr(event, "metadata_label", None) or (
        "CHAPTER 01 // THE PATTERN" if event.edit_type == "black_chapter_card" else "SOURCE // CONTEXT" if event.source_url else "QUOTED SLOGANS")
    source = html.escape(event.subtext_source or "")
    theme = getattr(event, "card_theme", "paper")
    image=''
    if event.evidence_image_path:
        path=Path(event.evidence_image_path)
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=event.evidence_image_sha256:
            raise PipelineError('Evidence image hash differs from publisher receipt')
        mime=mimetypes.guess_type(path.name)[0] or 'image/png'
        image='<div class="news-photo"><img src="data:'+mime+';base64,'+base64.b64encode(raw).decode()+'"></div>'
    publisher='<div class="masthead">'+html.escape(event.publisher or '')+'</div>' if event.publisher else ''
    return f'''<!doctype html><html><head><meta charset="utf-8"><style>
{_font_css()}
*{{box-sizing:border-box}}html,body{{margin:0;width:{width}px;height:{height}px;overflow:hidden;background:transparent}}
body{{font-family:Editorial,Devanagari,'Nirmala UI',sans-serif;-webkit-font-smoothing:antialiased;color:#141821}}
#entrance,#exit{{position:absolute;inset:0}} .surface{{position:absolute;left:18%;top:25%;width:64%;padding:48px 58px 44px;
background:#F8F6F0;border:1px solid rgba(255,255,255,.65);border-radius:8px;
box-shadow:0 25px 50px -12px rgba(0,0,0,.75),0 6px 15px rgba(0,0,0,.3);transform:perspective(1600px) rotateY(-1.2deg) rotateZ(-1.2deg)}}
.surface:after,body.black_chapter_card .grain:after{{content:'';position:absolute;inset:0;pointer-events:none;opacity:.04;
background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.9'/%3E%3C/svg%3E")}}
.meta{{font-family:Mono,monospace;font-size:17px;letter-spacing:2px;font-weight:600;color:#78633e;display:flex;align-items:center;gap:12px}}
.dot{{width:9px;height:9px;background:#c75335;border-radius:50%;box-shadow:0 0 0 5px rgba(197,83,53,.08)}}
.toprule{{height:1px;background:rgba(41,38,31,.22);margin:24px 0 28px}}
h1{{font-size:68px;line-height:1.2;letter-spacing:-2.5px;font-weight:800;margin:0 0 28px;overflow-wrap:anywhere}}
.source{{font:16px/1.65 Mono,Devanagari,monospace;color:#6b6255;white-space:pre-line;letter-spacing:.3px}}
mark{{background:none;position:relative;display:inline-block;color:inherit;padding:0 .06em;margin:0 -.06em}}
.sweep{{position:absolute;inset:10% -2% 2%;transform-origin:left;transform:scaleX(0);background:linear-gradient(104deg,rgba(250,204,21,0) .9%,rgba(250,204,21,.95) 2.4%,rgba(250,204,21,.85) 98%);z-index:0}}
.ink{{position:relative;z-index:1}}
body.glass .surface{{background:rgba(18,22,30,.88);backdrop-filter:blur(20px);color:#f7f5ef;border-color:rgba(255,255,255,.15)}}
body.glass .source{{color:#c8c6c2}}body.glass .meta{{color:#f1bc58}}
body.lower_third_citation .surface{{left:6%;top:auto;bottom:10%;width:78%;padding:25px 34px;border-radius:9px;
background:rgba(12,15,22,.85);backdrop-filter:blur(20px);border-left:5px solid #F59E0B;transform:none;
box-shadow:0 12px 40px rgba(0,0,0,.45),0 0 28px rgba(245,158,11,.08);color:white}}
body.lower_third_citation h1{{font-size:39px;letter-spacing:-.8px;margin:12px 0 0;line-height:1.25}}
body.lower_third_citation .meta{{color:#F59E0B;font-size:13px;letter-spacing:3px}}body.lower_third_citation .toprule,body.lower_third_citation .source{{display:none}}
body.black_chapter_card #exit{{background:radial-gradient(ellipse at 50% 44%,#0B0D13 0%,#040508 55%,#000 100%);color:#f9f6ed}}
body.black_chapter_card .surface{{background:none;border:0;box-shadow:none;transform:none;left:12%;width:76%;top:27%;padding:0;text-align:center}}
body.black_chapter_card .surface:after{{display:none}}
body.black_chapter_card .meta{{justify-content:center;font-size:18px;color:#c6a768;letter-spacing:4px}}
body.black_chapter_card .dot{{display:none}}body.black_chapter_card .toprule{{height:2px;width:320px;margin:32px auto 52px;background:linear-gradient(90deg,transparent,#d6b676,#fff4d7,#d6b676,transparent);transform:scaleX(0)}}
body.black_chapter_card h1{{font-size:91px;line-height:1.36;font-weight:800;letter-spacing:-2.8px;max-width:1400px;margin:auto}}
body.black_chapter_card .source{{color:#9b927f;font-size:17px;margin-top:44px;letter-spacing:3px}}
.word{{display:inline-block;opacity:0;filter:blur(6px);transform:translateY(12px)}}
.masthead{{font:800 37px Georgia,serif;letter-spacing:1px;padding-bottom:14px;border-bottom:3px double #26221c;margin-bottom:24px}}
.news-photo{{float:left;width:38%;margin:0 34px 15px 0}}.news-photo img{{display:block;width:100%;height:240px;object-fit:cover;border-radius:3px}}
body.evidence_card .surface{{left:12%;width:76%;top:17%;padding:32px 42px 34px}}
body.evidence_card h1{{font-size:45px;line-height:1.25;letter-spacing:-1.2px}}body.evidence_card .toprule{{margin:16px 0 22px}}
body.document_card .surface{{left:10%;width:80%;top:9%;padding:26px 40px;min-height:80%;transform:perspective(1800px) rotateZ(-.6deg)}}
body.document_card .news-photo{{float:right;width:45%;height:670px;margin:0 0 0 36px;background:#ece9e2}}
body.document_card .news-photo img{{height:100%;object-fit:contain}}body.document_card h1{{font-size:53px;line-height:1.35;letter-spacing:-1.3px}}
body.document_card .masthead{{font-size:29px}}body.document_card .source{{font-size:17px;line-height:1.8;white-space:pre-wrap}}
body.slogan_cards #exit{{background:radial-gradient(ellipse at 50% 44%,#10151f,#020306 85%)}}
body.slogan_cards .surface{{left:10%;width:80%;top:16%;padding:0;background:none;border:0;box-shadow:none;transform:none;color:#f8f6f0}}
body.slogan_cards .surface:after{{display:none}}body.slogan_cards .meta{{color:#f5ba54;font-size:16px}}
body.slogan_cards h1{{font-size:55px;line-height:1.55;letter-spacing:-.8px;margin:24px 0}}
.quote{{background:rgba(255,255,255,.05);border-left:3px solid #dfb154;border-radius:4px;padding:20px 30px;margin-bottom:25px;opacity:0}}
.quote-number{{font:14px Mono;letter-spacing:3px;color:#d4b57b;margin-bottom:12px}}
body.side_by_side h1{{display:grid;grid-template-columns:1fr 1fr;gap:26px;font-size:57px}}
body.side_by_side .quote{{border-left:0;border-top:3px solid #dfb154;padding:30px;align-self:start}}
body.slogan_cards .source{{color:#b9b4a9;font-size:14px}}
</style></head><body class="{event.edit_type} {theme}"><div id="entrance"><div id="exit" class="grain"><section class="surface">
{publisher}<div class="meta"><span class="dot"></span>{html.escape(metadata)}</div><div class="toprule"></div>{image}<h1>{safe}</h1><div class="source">{source}</div>
</section></div></div><script>(()=>{{
const d={json.dumps(data)};const animations=[];
function anim(el,k,o){{const a=el.animate(k,{{fill:'both',...o}});a.pause();animations.push(a);}}
anim(document.querySelector('#entrance'),[{{opacity:0,transform:'translateY(28px)'}},{{opacity:1,transform:'translateY(0)'}}],{{duration:d.kind==='lower_third_citation'?200:180,easing:'cubic-bezier(0.16,1,0.30,1)'}});
anim(document.querySelector('#exit'),[{{opacity:1}},{{opacity:0}}],{{duration:120,delay:d.duration-120}});
document.querySelectorAll('.sweep').forEach(el=>anim(el,[{{transform:'scaleX(0)'}},{{transform:'scaleX(1)'}}],{{duration:260,delay:120,easing:'cubic-bezier(.16,1,.3,1)'}}));
if(d.kind==='slogan_cards'){{document.body.classList.add('{event.quote_layout}');let offset=0;
document.querySelectorAll('.quote').forEach((el,i)=>{{anim(el,[{{opacity:0,transform:'translateY(16px)'}},{{opacity:1,transform:'translateY(0)'}}],{{duration:200,delay:d.times[offset]??0,easing:'cubic-bezier(.16,1,.3,1)'}});offset+=el.querySelectorAll('.word').length;}});}}
if(d.kind==='black_chapter_card'||d.kind==='slogan_cards'){{anim(document.querySelector('.toprule'),[{{transform:'scaleX(0)'}},{{transform:'scaleX(1)'}}],{{duration:380,delay:80,easing:'cubic-bezier(.16,1,.3,1)'}});
document.querySelectorAll('.word').forEach((el,i)=>anim(el,[{{opacity:0,filter:'blur(6px)',transform:'translateY(12px)'}},{{opacity:1,filter:'blur(0)',transform:'translateY(0)'}}],{{duration:180,delay:d.times[i]??(220+i*120),easing:'cubic-bezier(.22,1,.36,1)'}}));}}
window.seek=async ms=>{{await Promise.all(animations.map(a=>a.ready));animations.forEach(a=>a.currentTime=ms);
document.querySelectorAll('.word,.sweep,.toprule').forEach(e=>getComputedStyle(e).transform);
await new Promise(resolve=>requestAnimationFrame(resolve));}};
}})();</script></body></html>'''


class HTMLOverlayRenderer:
    def __init__(self, cache_root: Path):
        self.root = cache_root
        self.root.mkdir(parents=True, exist_ok=True)

    def prepare(self, events, width, height, fps, words=()):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise PipelineError("HTML documentary overlays require the edit extra and Playwright Chromium") from exc
        result, layouts = {}, []
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1,
                                    reduced_motion="no-preference")
            try:
                for event in events:
                    if not event.is_graphic:
                        continue
                    times, timing = kinetic_word_times(event, words)
                    document = build_html(event, width, height, times)
                    digest = hashlib.sha256((VERSION + document + str(fps)).encode()).hexdigest()[:20]
                    directory = self.root / (event.event_id + "-" + digest)
                    directory.mkdir(exist_ok=True)
                    count = math.ceil((event.end_sec - event.start_sec) * fps)
                    marker = directory / "complete.json"
                    if not (marker.is_file() and len(list(directory.glob('frame-*.png'))) == count):
                        (directory / "index.html").write_text(document, encoding="utf-8")
                        page.set_content(document, wait_until="load")
                        page.evaluate("document.fonts.ready")
                        layout = page.locator('.surface').bounding_box()
                        overflow = page.evaluate("() => {const e=document.querySelector('.surface');return e.scrollHeight>e.clientHeight+2 || e.getBoundingClientRect().bottom>innerHeight*.94}")
                        if overflow:
                            raise PipelineError(f"HTML card {event.event_id} exceeds its safe area")
                        settled = max(.46, max(times, default=0) + .2) if event.edit_type in {"black_chapter_card","slogan_cards"} else .2 if event.edit_type == "lower_third_citation" else .38
                        static = directory / "settled.png"
                        for i in range(count):
                            elapsed = i / fps
                            target = directory / f"frame-{i:05}.png"
                            if settled + 1/fps <= elapsed < (event.end_sec-event.start_sec)-.12:
                                # All WAAPI effects are paused/settled: these PNGs are pixel-identical.
                                # Hard-link the actual browser frame, preserving a full seekable sequence.
                                if not static.exists():
                                    page.evaluate("(ms) => window.seek(ms)", elapsed * 1000)
                                    page.screenshot(path=str(static), omit_background=True, animations="allow")
                                target.unlink(missing_ok=True)
                                try:
                                    target.hardlink_to(static)
                                except OSError:
                                    shutil.copyfile(static, target)
                            else:
                                page.evaluate("(ms) => window.seek(ms)", elapsed * 1000)
                                page.screenshot(path=str(target), omit_background=True,
                                                animations="allow", timeout=30000)
                        marker.write_text(json.dumps({"frames": count, "fps": fps, "layout": layout, **timing}, indent=2))
                    info = json.loads(marker.read_text())
                    result[event.event_id] = {"directory": directory, "frames": count, "fps": fps}
                    layouts.append({"event_id": event.event_id, "renderer": "Chromium WebAnimations transparent PNG",
                                    "alpha_mode": "straight PNG converted to premultiplied for composition", **info})
                    print(f"HTML overlay ready: {event.event_id} ({count} real PNG frames)", flush=True)
            finally:
                browser.close()
        return result, layouts


def composite_premultiplied(background, rgba):
    """PNG is straight BGRA; convert RGB to premultiplied before the over operator."""
    if rgba.shape[2] == 3:
        rgba = np.dstack([rgba, np.full(rgba.shape[:2], 255, np.uint8)])
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255
    premultiplied = rgba[:, :, :3].astype(np.float32) * alpha
    return np.clip(premultiplied + background.astype(np.float32) * (1 - alpha), 0, 255).astype(np.uint8)


def composite_overlay(frame, event, time, sequence):
    from .cv_detectors import require_cv
    cv2 = require_cv()
    h, w = frame.shape[:2]
    elapsed = max(0, time - event.start_sec)
    i = min(sequence['frames'] - 1, max(0, round(elapsed * sequence['fps'])))
    rgba = cv2.imread(str(sequence['directory'] / f"frame-{i:05}.png"), cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.shape[:2] != (h, w) or rgba.shape[2] not in (3, 4):
        raise PipelineError("Missing or invalid transparent overlay frame")
    if rgba.shape[2] == 3:
        rgba = cv2.cvtColor(rgba, cv2.COLOR_BGR2BGRA)
    if event.edit_type in {"evidence_card","document_card"}:
        from .visual_pacing import cubic_bezier_y_from_x
        strength = min(cubic_bezier_y_from_x(elapsed / .18), max(0, (event.end_sec - time) / .12))
        zoom = 1 + .04 * min(1, elapsed / max(.01, event.end_sec - event.start_sec))
        cw, ch = round(w / zoom), round(h / zoom)
        crop = frame[(h-ch)//2:(h-ch)//2+ch, (w-cw)//2:(w-cw)//2+cw]
        treated = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LANCZOS4)
        # Half-resolution sigma=9 equals sigma=18 in the source pixel grid.
        treated = cv2.resize(cv2.GaussianBlur(cv2.resize(treated, (w//2, h//2)), (0, 0), 9), (w, h))
        y, x = np.ogrid[-1:1:h*1j, -1:1:w*1j]
        vignette = np.clip(.69 - .34 * (x*x + y*y), .35, .69).astype(np.float32)
        treated = (treated.astype(np.float32) * vignette[:, :, None]).astype(np.uint8)
        frame = cv2.addWeighted(treated, strength, frame, 1-strength, 0)
    elif event.edit_type == "lower_third_citation":
        # Chromium's transparent page cannot sample the A-roll for backdrop-filter.
        # Supply the actual blurred pixels underneath its frosted alpha region here.
        blurred = cv2.resize(cv2.GaussianBlur(cv2.resize(frame, (w//2, h//2)), (0, 0), 6), (w, h))
        mask = np.clip((rgba[:, :, 3:4].astype(np.float32) / 255 - .35) / .5, 0, 1)
        frame = np.clip(frame * (1-mask) + blurred * mask, 0, 255).astype(np.uint8)
    return composite_premultiplied(frame, rgba)
