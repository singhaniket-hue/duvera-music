"""Synthetic AV fixtures are disabled. Use the real-footage documentary demo helper."""
from src.media import PipelineError

def create(*args, **kwargs):
    raise PipelineError('Synthetic video/tone fixtures are disabled; supply real --video and --transcript to scripts/create_documentary_demo.py')

if __name__ == '__main__':
    raise SystemExit('Synthetic AV demo disabled. Use scripts/create_documentary_demo.py --video REAL.mp4 --transcript transcript.json --dir OUTPUT')
