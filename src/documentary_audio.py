"""Post-normalization documentary accents and a mandatory encoded-program audibility gate."""
from __future__ import annotations

import json
import math

import numpy as np
from scipy.signal import butter, sosfilt

from .media import SR, PipelineError, ffmpeg, extract_audio, loudness, normalize, read_audio, write_audio


def apply_pre_silence_reverb_tails(music,sheet,receipt_path):
    """Bounded reflections of the real bed, wholly BEFORE protected chapter/thesis speech."""
    import hashlib
    def digest():
        return hashlib.sha256(music.read_bytes()).hexdigest()
    original=digest()
    if receipt_path.exists():
        cached=json.loads(receipt_path.read_text())
        if cached.get('output_sha256')==original:
            return cached
    audio,_=read_audio(music)
    windows=[]
    for window in sheet.silence_windows:
        if window.kind=='archival' or not window.accent_event_ids:
            continue
        end=round(window.start_sec*SR)
        length=round(min(.2,window.fade_sec)*SR)
        start=max(0,end-length)
        dry=audio[start:end].copy()
        tail=np.zeros_like(dry)
        for delay,gain in ((.013,.22),(.023,.16),(.037,.11),(.053,.07)):
            lag=round(delay*SR)
            if lag<len(tail):
                tail[lag:]+=dry[:-lag]*gain
        if len(tail):
            tail*=np.linspace(1,0,len(tail),dtype=np.float32)[:,None]**2
            audio[start:end]+=tail
        windows.append({'silence_start_sec':window.start_sec,'silence_end_sec':window.end_sec,
                        'tail_ms':length/SR*1000,'source':'recorded music reflections only'})
    # All protected windows still have hard sample-zero added music.
    for window in sheet.silence_windows:
        audio[round(window.start_sec*SR):round(window.end_sec*SR)]=0
    write_audio(music,audio)
    report={'input_sha256':original,'output_sha256':digest(),'windows':windows,
            'method':'real-source delayed early reflections decaying before the protected boundary'}
    receipt_path.write_text(json.dumps(report,indent=2))
    return report


def build_documentary_sfx_stem(pacing,sheet,store,dialogue,root,config):
    from .dsp_mixer import render_effects
    expected=[e for e in pacing.effects if e.required]
    result=render_effects(pacing,sheet,store,dialogue,root,config)
    placed={e['event_id'] for e in result['effects'] if e.get('status')=='placed'}
    if any(e.event_id not in placed for e in expected):
        raise PipelineError('Required documentary SFX missing from actual stem')
    return result


def mux_final_documentary_master(dialogue,music,sfx,root,config):
    """Normalize dialogue/music alone; add accents at +3dB; only peak-limit and constant-gain after."""
    base=root/'dialogue_music_float.wav'
    normalized=root/'dialogue_music_normalized.wav'
    ffmpeg('-i',dialogue,'-i',music,'-filter_complex','[0:a][1:a]amix=inputs=2:duration=first:normalize=0',
           '-ar',SR,'-c:a','pcm_f32le',base)
    normalize(base,normalized,config.target_lufs,config.true_peak_db)
    combined=root/'post_sfx_limited.wav'
    limiter=f'alimiter=limit={10**(-1.8/20)}:attack=2:release=70:level=false:latency=true'
    ffmpeg('-i',normalized,'-i',sfx,'-filter_complex',
        f'[1:a]volume=3dB[fx];[0:a][fx]amix=inputs=2:duration=first:normalize=0,{limiter}[m]',
        '-map','[m]','-ar',SR,'-c:a','pcm_s24le',combined)
    measured=loudness(combined)
    gain=min(config.target_lufs-measured['input_i'],config.true_peak_db-measured['input_tp']-.2)
    master=root/'master.wav'
    ffmpeg('-i',combined,'-af',f'volume={gain}dB','-ar',SR,'-c:a','pcm_s24le',master)
    # Counterfactual uses the same limiter/gain/codec, with only SFX omitted.
    reference=root/'encoded_without_sfx.m4a'
    ffmpeg('-i',normalized,'-af',limiter+f',volume={gain}dB','-ar',SR,'-c:a','aac','-b:a','192k',reference)
    # Conservative music-bed upper bound: largest actual normalization gain over 10ms blocks.
    import soundfile as sf
    raw,_=sf.read(base,dtype='float32',always_2d=True)
    norm,_=sf.read(normalized,dtype='float32',always_2d=True)
    n=min(len(raw),len(norm))//480
    a=np.sqrt(np.mean(raw[:n*480].reshape(n,480,2).astype(float)**2,axis=(1,2)))
    b=np.sqrt(np.mean(norm[:n*480].reshape(n,480,2).astype(float)**2,axis=(1,2)))
    max_gain=float(np.max(b[a>1e-4]/a[a>1e-4]))
    bed,_=read_audio(music)
    write_audio(root/'audibility_music_upper_bound.wav',bed*max_gain*10**(gain/20))
    report={'sfx_gain_relative_to_normalized_base_db':3,'final_constant_gain_db':gain,
            'music_normalization_upper_bound_gain':max_gain,
            'sfx_after_loudnorm':True,'post_sfx_processing':'peak limiter then constant gain; no loudnorm',
            'counterfactual':reference.name}
    (root/'documentary_gain_staging.json').write_text(json.dumps(report,indent=2))
    for path in (base,combined):
        path.unlink(missing_ok=True)
    return master,report


def _band(samples,sub,whoosh=False):
    spec=[35,150] if sub else 300 if whoosh else 1800
    return sosfilt(butter(3,spec,btype='bandpass' if sub else 'highpass',fs=SR,output='sos'),samples,axis=0)


def _energy(samples):
    return 10*math.log10(max(float(np.mean(np.asarray(samples,dtype=float)**2)),1e-14))


def verify_final_sfx_audibility(video,root,pacing):
    decoded=root/'encoded_program.wav'
    reference=root/'encoded_reference.wav'
    extract_audio(video,decoded)
    extract_audio(root/'encoded_without_sfx.m4a',reference)
    final,_=read_audio(decoded)
    ref,_=read_audio(reference)
    bed,_=read_audio(root/'audibility_music_upper_bound.wav')
    count=min(len(final),len(ref),len(bed))
    final,ref,bed=final[:count],ref[:count],bed[:count]
    events=[]
    for item in pacing.effects:
        if item.tier!='accent':
            continue
        target=item.visual_time_sec+item.av_offset_ms/1000
        a,b=max(0,round((target-.05)*SR)),min(count,round((target+.25)*SR))
        # Background includes the target and its ±400ms surrounding bed, conservatively.
        left,right=max(0,a-round(.4*SR)),min(count,b+round(.4*SR))
        # Filter only the measured neighbourhood, with a 250ms IIR warm-up. This
        # bounds memory for long films instead of allocating nine full-film float64 buses.
        begin=max(0,left-round(.25*SR))
        sub,whoosh=item.category=='sub_drop',item.category in {'whoosh','whoosh_soft','riser'}
        program=_band(final[begin:right],sub,whoosh)
        contribution=_band(final[begin:right]-ref[begin:right],sub,whoosh)
        background=_band(bed[begin:right],sub,whoosh)
        bed_db=max(_energy(background[a-begin:b-begin]),_energy(background[left-begin:right-begin]))
        program_db=_energy(program[a-begin:b-begin])
        contribution_db=_energy(contribution[a-begin:b-begin])
        margin=min(program_db,contribution_db)-bed_db
        events.append({'event_id':item.event_id,'target_sec':target,'window_start_sec':a/SR,'window_end_sec':b/SR,
            'band':'35-150Hz' if sub else 'highpass300Hz' if whoosh else 'highpass1800Hz',
            'encoded_program_band_rms_dbfs':program_db,'encoded_sfx_contribution_band_rms_dbfs':contribution_db,
            'conservative_surrounding_music_band_rms_dbfs':bed_db,'margin_db':margin,'required_margin_db':8,
            'passed':margin>=8})
    report={'passed':bool(events) and all(e['passed'] for e in events),'events':events,
            'method':'Decoded final MP4 band RMS and final-minus-matched-AAC-counterfactual contribution; both >=8dB above conservative music-bed upper bound',
            'limitation':'Technical audibility against music; does not substitute for listening or measure masking by speech'}
    (root/'final_sfx_audibility.json').write_text(json.dumps(report,indent=2))
    if not report['passed']:
        raise PipelineError('Encoded MP4 SFX audibility gate failed; inspect final_sfx_audibility.json')
    return report
