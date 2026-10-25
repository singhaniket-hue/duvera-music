"""Source-clock false-start removal and measured non-speech gap cuts before planning.

ASR is fallible: decisions retain tokens, scores and acoustic evidence for review.
No media or audio is fabricated by this module.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

import numpy as np

from .media import PipelineError, SR


def token(text):
    return re.sub(r'[^\w\u0900-\u097f]', '', text.casefold()).replace('।', '')


def detect_retakes(transcript, audio, inserts=()):
    if transcript.timing_source != 'asr' or not transcript.words:
        raise PipelineError('Retake auto-cutter requires faster-whisper word timestamps on raw media first')
    words = transcript.words
    cuts, decisions = [], []
    candidates=[]
    for i, first in enumerate(words[:-3]):
        best = None
        for j in range(i + 2, len(words) - 1):
            if words[j].start - first.start > 4.0:
                break
            # Avoid one-word stutters and common disconnected function-word matches.
            if SequenceMatcher(None,token(first.word),token(words[j].word)).ratio()<.75:
                continue
            n = 0
            for a, b in zip(words[i:j], words[j:]):
                if SequenceMatcher(None, token(a.word), token(b.word)).ratio() < .75:
                    break
                n += 1
            if n < 2:
                continue
            run=longest=0
            for left,right in zip(words[i:i+n],words[j:j+n]):
                run=run+1 if token(left.word)==token(right.word) else 0
                longest=max(longest,run)
            if longest<2:
                # मर्दों में / पर्दों में is a rhyme, not a restarted clause.
                continue
            if n==2 and token(first.word)!=token(words[j].word):
                # Short rhyming phrases (दर्द है / मर्द है) are not false starts.
                # Fuzzy first-token matches need a longer corroborating prefix.
                continue
            if j-i-n > max(2,n//2):
                # Two sentences can legitimately share "the boys", "we are", etc.
                # A restart resumes a short prefix, not a completed intervening sentence.
                continue
            a = ' '.join(token(w.word) for w in words[i:i+n])
            b = ' '.join(token(w.word) for w in words[j:j+n])
            score = SequenceMatcher(None, a, b).ratio()
            # An immediate identical short pair is often an intentional list/rhetorical repeat.
            # Require a changed continuation, intervening gap or at least three matching tokens.
            continuation_changed = (i+n < j and j+n < len(words) and
                                    token(words[i+n].word) != token(words[j+n].word))
            gap = words[j].start - words[j-1].end
            if score >= .75 and (continuation_changed or gap > .18 or n >= 3):
                candidate = (n, score, j, a, b)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
        if best:
            n, score, j, a, b = best
            start, end = first.start, words[j].start
            if i:
                start=max(start,words[i-1].end)
            # Keep a little quiet lead at the retained restart, without keeping the false words.
            if i and first.start - words[i-1].end > .02:
                start = first.start - min(.04, (first.start-words[i-1].end)/2)
            decision = dict(start_sec=start, end_sec=end, reason='repeated_false_start',
                            first_tokens=a, retained_tokens=b, matched_words=n, similarity=score)
            if not any(start < w.end_sec and end > w.start_sec for w in inserts):
                candidates.append(decision)
            else:
                decisions.append({**decision, 'accepted':False, 'reason':'protected_archival'})
    # Resolve overlapping hypotheses longest-prefix first. An earlier two-word
    # suffix of a complete sentence must not eat the start of a later full restart.
    for candidate in sorted(candidates,key=lambda c:(-c['matched_words'],-c['similarity'],c['start_sec'])):
        if not any(candidate['start_sec']<c['end_sec'] and candidate['end_sec']>c['start_sec'] for c in cuts):
            cuts.append(candidate)
    # Gaps are only cut when the waveform supports quiet air or a compact isolated burst.
    if audio is not None:
        for left, right in zip(words, words[1:]):
            a, b = left.end, right.start
            if b-a <= .45 or any(a < c['end_sec'] and b > c['start_sec'] for c in cuts):
                continue
            if any(a < w.end_sec and b > w.start_sec for w in inserts):
                continue
            chunk = audio[round((a+.03)*SR):round((b-.03)*SR)]
            count = len(chunk)//480
            if count < 5:
                continue
            rms = np.sqrt(np.mean(chunk[:count*480].astype(float).reshape(count,480,2)**2,axis=(1,2)))
            db = 20*np.log10(np.maximum(rms,1e-10))
            floor, crest = float(np.percentile(db,20)), float(db.max())
            active_fraction = float(np.mean(db > max(-36, floor+14)))
            quiet = b-a > .55 and crest < -36
            burst = floor < -40 and crest > floor+18 and .02 <= active_fraction < .55
            decision = dict(start_sec=a+.09,end_sec=b-.09,reason='non_speech_impulse' if burst else 'dead_air',
                gap_start_sec=a,gap_end_sec=b,retained_gap_ms=180,noise_floor_dbfs=floor,
                max_10ms_rms_dbfs=crest,active_fraction=active_fraction,accepted=quiet or burst,
                classification='acoustic non-speech-gap proxy; not a medical cough classifier')
            decisions.append(decision)
            if quiet or burst:
                cuts.append(decision)
    cuts.sort(key=lambda c:c['start_sec'])
    merged=[]
    for cut in cuts:
        if merged and cut['start_sec'] < merged[-1]['end_sec']:
            merged[-1]['end_sec']=max(merged[-1]['end_sec'],cut['end_sec'])
        else:
            merged.append(cut)
    return merged, decisions


def splice_switches(mapping):
    """Scale changes exactly on the first retained frame after every source discontinuity."""
    return [{'frame':r.output_start_frame,'time_sec':r.output_start_frame/mapping.fps,
             'scale':1.18 if i % 2 else 1.0,
             'source_restart_sec':r.source_start_frame/mapping.fps}
            for i,r in enumerate(mapping.keep_ranges) if i]
