"""No synthetic AV: browser contracts, actual acoustic cache, and supplied real footage."""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError

from src.documentary_schema import VisualEditEvent
from src.html_overlay_renderer import build_html, kinetic_word_times
from src.media import PipelineError
from src.models import RunConfig
from src.procedural_sfx import synthesize
from src.sfx_library import SFXAssetLibrary, analyze_and_shape_sfx_asset

ROOT = Path(__file__).resolve().parents[1]


class ProductionContracts(unittest.TestCase):
    def test_synthetic_paths_fail_closed(self):
        with self.assertRaises(ValidationError):
            RunConfig(script_path=Path('script.txt'), procedural_sfx=True)
        with self.assertRaisesRegex(PipelineError, 'disabled'):
            synthesize('sub_drop', 1, Path('never-created.wav'))
        from scripts.create_documentary_demo import create
        with self.assertRaisesRegex(PipelineError, 'Real source'):
            create(Path('never-created'))

    def test_preset_uses_real_asset_path(self):
        config = RunConfig(preset='social_commentary', script_path=Path('script.txt'))
        self.assertFalse(config.procedural_sfx)
        self.assertEqual(config.documentary_edit_mode, 'plan')

    def test_pending_generation_never_retries(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'sub_drop.pending.json').write_text('{"status":"pending"}')
            with patch.dict(os.environ, {'ELEVENLABS_API_KEY':'not-a-real-key'}), \
                    patch('src.sfx_library.urlopen', side_effect=AssertionError('duplicate paid call')):
                with self.assertRaisesRegex(PipelineError, 'Uncertain/active'):
                    SFXAssetLibrary(root).get('sub_drop', allow_generation=True)


class BrowserContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from playwright.sync_api import sync_playwright
        cls.p = sync_playwright().start()
        cls.browser = cls.p.chromium.launch()
        cls.page = cls.browser.new_page(viewport={'width':1920,'height':1080})

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.p.stop()

    def load(self, event, times=()):
        self.page.set_content(build_html(event,1920,1080,times),wait_until='load')
        self.page.evaluate('document.fonts.ready')

    def seek(self, ms):
        self.page.evaluate('(ms)=>window.seek(ms)',ms)

    def test_multiple_cards_reuse_page_and_highlight_sweep_is_seekable(self):
        errors=[]
        self.page.on('pageerror',lambda e: errors.append(str(e)))
        for _ in range(2):
            event=VisualEditEvent(start_sec=0,end_sec=8,edit_type='evidence_card',headline_text='15% vs 2%')
            self.load(event)
            self.seek(100)
            before=self.page.locator('.sweep').first.evaluate('(e)=>getComputedStyle(e).transform')
            self.seek(500)
            after=self.page.locator('.sweep').first.evaluate('(e)=>getComputedStyle(e).transform')
            self.assertNotEqual(before,after)
            self.assertTrue(after.startswith('matrix(1,'))
        self.assertFalse(errors)

    def test_word_reveal_follows_each_asr_word_and_hindi_shapes(self):
        event=VisualEditEvent(start_sec=7.5,end_sec=15.5,edit_type='black_chapter_card',headline_text='एक कचोड़ी, दो समोसा।')
        words=[{'start':7.78,'end':8.16,'word':'एक'}, {'start':8.16,'end':8.64,'word':'कचोड़ी,'},
               {'start':8.76,'end':8.9,'word':'दो'}, {'start':8.9,'end':9.4,'word':'समोसा,'}]
        times,info=kinetic_word_times(event,words)
        self.assertEqual(info['matched_words'],4)
        self.load(event,times)
        self.seek(500)
        opacity=self.page.locator('.word').evaluate_all('(es)=>es.map(e=>Number(getComputedStyle(e).opacity))')
        self.assertGreater(opacity[0],.99)
        self.assertEqual(opacity[1:], [0,0,0])
        self.seek(2400)
        self.assertEqual(self.page.locator('.word').evaluate_all('(es)=>es.map(e=>Number(getComputedStyle(e).opacity))'),[1,1,1,1])
        self.assertTrue(self.page.evaluate("document.fonts.check('90px Devanagari')"))

    def test_lower_third_keeps_caption_safe_area_and_alpha(self):
        event=VisualEditEvent(start_sec=0,end_sec=6,edit_type='lower_third_citation',headline_text='Real source citation')
        self.load(event)
        self.seek(600)
        bounds=self.page.locator('.surface').bounding_box()
        self.assertAlmostEqual(bounds['x'],1920*.06,delta=1)
        self.assertAlmostEqual(bounds['y']+bounds['height'],1080*.9,delta=1)
        import cv2
        png=self.page.screenshot(omit_background=True)
        image=cv2.imdecode(np.frombuffer(png,np.uint8),cv2.IMREAD_UNCHANGED)
        self.assertEqual(image.shape[2],4)
        self.assertEqual(int(image[0,0,3]),0)


@unittest.skipUnless((ROOT/'assets/sfx_cache/sub_drop.wav').exists(), 'Requires an actual acoustic cache')
class RealAcousticCache(unittest.TestCase):
    def test_real_cache_hit_does_not_call_provider_and_crest_is_measured(self):
        library=SFXAssetLibrary()
        with patch('src.sfx_library.urlopen',side_effect=AssertionError('cache hit called API')):
            source=library.get('sub_drop',allow_generation=True)
        info=analyze_and_shape_sfx_asset(source,'sub_drop')
        self.assertGreaterEqual(info['dominant_peak_time_s'],0)
        self.assertLess(info['dominant_peak_time_s'],info['duration_seconds'])
        provenance=json.loads(source.with_suffix('.json').read_text())
        self.assertEqual(provenance['source'],'elevenlabs_sound_generation')
        self.assertFalse(provenance['synthetic_generator_used'])

    def test_cache_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            source=ROOT/'assets/sfx_cache/sub_drop.wav'
            shutil.copyfile(source,Path(temp)/source.name)
            (Path(temp)/'sub_drop.json').write_text('{"sha256":"invalid"}')
            with self.assertRaisesRegex(PipelineError,'changed'):
                SFXAssetLibrary(Path(temp)).get('sub_drop')


@unittest.skipUnless(os.getenv('DUVERA_REAL_DEMO'), 'Set DUVERA_REAL_DEMO to the real rendered clip project')
class ActualFootageIntegration(unittest.TestCase):
    def test_delivered_real_media_and_independent_qc(self):
        root=Path(os.environ['DUVERA_REAL_DEMO'])
        qc=json.loads((root/'delivery-qc.json').read_text())
        self.assertTrue(qc['passed'])
        self.assertEqual(qc['synthetic_audio_generators_used'],0)
        self.assertTrue(qc['real_source_video'].endswith('01_Protest_and_response.mp4'))
        self.assertEqual(len(qc['sfx_waveform_checks']),3)


if __name__=='__main__':
    unittest.main()
