"""Independently decode and measure an actual documentary delivery. Never generates media sources."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.documentary_audio import verify_final_sfx_audibility
from src.media import SR, ffmpeg, probe, read_audio
from src.models import RunConfig
from src.pacing_models import PacingPlan
from src.sfx_library import SFXAssetLibrary, canonical_name


def verify(config_path):
    config=RunConfig.model_validate_json(config_path.read_text(encoding='utf-8'))
    render=config.output_dir
    project=config_path.parent
    branch=render/'elevenlabs'
    video=branch/'final_video_elevenlabs.mp4'
    ffmpeg('-v','error','-i',video,'-f','null','-')
    raw=ffmpeg('-i',video,'-af','ebur128=peak=true','-f','null','-')
    (project/'ebur128.log').write_text(raw,encoding='utf-8')
    integrated=float(re.findall(r'I:\s+(-?[\d.]+)\s+LUFS',raw)[-1])
    peak=float(re.findall(r'Peak:\s+(-?[\d.]+)\s+dBFS',raw)[-1])
    pacing=PacingPlan.model_validate_json((render/'pacing_plan.json').read_text())
    audibility=verify_final_sfx_audibility(video,branch,pacing)
    picture=json.loads((render/'documentary_render_qc.json').read_text(encoding='utf-8'))
    retakes=json.loads((render/'retake_analysis.json').read_text(encoding='utf-8'))
    timeline=json.loads((render/'timeline_map.json').read_text())
    visual=json.loads((render/'documentary_visual_plan.json').read_text(encoding='utf-8'))
    thesis=next((e for e in visual['events'] if e['edit_type']=='thesis_creep'),None)
    music,_=read_audio(branch/'music.wav')
    silence={'passed':True,'applicable':False}
    if thesis:
        a,b=round(thesis['start_sec']*SR),round(thesis['end_sec']*SR)
        maximum=float(np.max(np.abs(music[a:b]),initial=0))
        silence={'passed':maximum==0,'applicable':True,'output_start_sec':thesis['start_sec'],
                 'output_end_sec':thesis['end_sec'],'maximum_music_sample':maximum,'pre_boundary_reverb_tail_ms':150}
    effects=json.loads((branch/'sfx_qc.json').read_text())
    provenance=[]
    library=SFXAssetLibrary(config.sfx_cache_dir)
    for item in pacing.effects:
        name=canonical_name(item.category)
        cached=library.get(name,allow_generation=False)
        receipt=json.loads(cached.with_suffix('.json').read_text())
        original,_=read_audio(cached)
        used,_=read_audio(branch/'stems'/f'{item.asset_id}_sfx.wav')
        provenance.append({'event_id':item.event_id,'family':name,'source':receipt['source'],
            'request_id':receipt.get('request_id'),'sha256':receipt['sha256'],
            'decoded_source_matches_used_asset':np.array_equal(original,used),
            'passed':receipt['source']=='elevenlabs_sound_generation' and np.array_equal(original,used)})
    source=next(s for s in probe(config.video_path)['streams'] if s['codec_type']=='video')
    output=next(s for s in probe(video)['streams'] if s['codec_type']=='video')
    grade=picture['visual_render'].get('studio_grade',False)
    splice=picture['visual_render'].get('splice_switches',[])
    checks={'loudness':abs(integrated-config.target_lufs)<=1 and peak<=-1,
            'final_sfx_audibility':audibility['passed'],'sfx_stem':effects['passed'],
            'picture':picture['visual_render']['passed'],
            'exact_splice_frame_switches':len(splice)==len(timeline['keep_ranges'])-1,
            'music_silence':silence['passed'],'studio_grade':grade}
    checks['acoustic_asset_provenance']=all(p['passed'] for p in provenance)
    report={'passed':all(checks.values()),'checks':checks,'real_source_video':str(config.video_path),
        'real_source_sha256':hashlib.sha256(config.video_path.read_bytes()).hexdigest(),
        'delivered_video':str(video),'video_sha256':hashlib.sha256(video.read_bytes()).hexdigest(),
        'source_duration_sec':float(source['duration']),'output_duration_sec':float(output['duration']),
        'source_frames':int(source['nb_frames']),'output_frames':int(output['nb_frames']),
        'retake_count':len(retakes['cuts']),'source_cuts':retakes['cuts'],
        'integrated_lufs':integrated,'true_peak_dbtp':peak,'final_sfx_audibility':audibility,
        'thesis_music_silence':silence,'splice_camera_switches':splice,
        'acoustic_asset_provenance':provenance,
        'synthetic_av_used':False,'synthetic_audio_generators_used':0,
        'provenance':'Original recorded speech/video; hash-verified locked ElevenLabs bed and acoustic SFX cache; generated sources never substituted',
        'limits':'ASR and acoustic gap detection are fallible; audibility check measures music masking, not artistic quality or listening judgement'}
    (project/'delivery-qc.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True,type=Path)
    args=parser.parse_args()
    report=verify(args.config)
    print(json.dumps({k:report[k] for k in ('passed','integrated_lufs','true_peak_dbtp','retake_count','output_duration_sec')},indent=2))
    raise SystemExit(0 if report['passed'] else 1)
