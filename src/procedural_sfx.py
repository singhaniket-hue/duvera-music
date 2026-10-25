"""Compatibility shim: synthetic production audio is permanently disabled."""
from .media import PipelineError

def synthesize(*args, **kwargs):
    raise PipelineError('Synthetic SFX are disabled; use acoustic SFXAssetLibrary or imported recordings.')

def procedural_jobs(pacing, config):
    if config.procedural_sfx:
        raise PipelineError('--procedural-sfx is disabled; use cached acoustic/provider assets')
    return {}
