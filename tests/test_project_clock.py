"""Integer-frame multi-part pause/resume contracts; no synthetic audiovisual fixtures."""
import unittest
import os
import json
import tempfile
from pathlib import Path

from src.production_project import remap_part_time, assembly_segments


class AssemblyClock(unittest.TestCase):
    def test_measured_music_loop_does_not_claim_verified_bar_phase(self):
        from src.rhythm import plan_loop
        loop=plan_loop(range(480000),{},explicit={'start_sec':1,'end_sec':8,
            'method':'measured_active_core','beat_quantized':False,'bar_phase_verified':False})
        self.assertEqual(loop['method'],'measured_active_core')
        self.assertFalse(loop['beat_quantized'])
        self.assertFalse(loop['bar_phase_verified'])

    def test_near_end_insert_absorbs_handle_without_empty_camera_flash(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            mapping={'rate_num':60,'rate_den':1,'source_frames':100,'keep_ranges':[
                {'source_start_frame':0,'source_end_frame':100,'output_start_frame':0}]}
            (root/'timeline_map.json').write_text(json.dumps(mapping))
            receipts=[{'root':str(root),'frames':100,'picture':'actual-source-path','dialogue':'actual-dialogue-path'}]
            parts=[{'id':'one'}]
            inserts=[{'part_id':'one','at_source_sec':1.6,'duration_sec':1}]
            segments,frames=assembly_segments(receipts,parts,inserts)
            self.assertEqual(frames,160)
            self.assertEqual([s['kind'] for s in segments],['camera','archive'])
            self.assertEqual(segments[0]['source_end_frame'],100)
            self.assertEqual(remap_part_time('one',100/60,segments),160/60)

    def test_pause_resume_shift_applies_at_exact_insert_boundary(self):
        segments=[{'kind':'camera','part_id':'one','source_start_frame':0,'source_end_frame':600,'output_start_frame':0},
                  {'kind':'archive','frames':120,'output_start_frame':600},
                  {'kind':'camera','part_id':'one','source_start_frame':600,'source_end_frame':1200,'output_start_frame':720},
                  {'kind':'camera','part_id':'two','source_start_frame':0,'source_end_frame':600,'output_start_frame':1320}]
        self.assertEqual(remap_part_time('one',9,segments),9)
        self.assertEqual(remap_part_time('one',10,segments),12)
        self.assertEqual(remap_part_time('one',20,segments),22)
        self.assertEqual(remap_part_time('two',0,segments),22)

    @unittest.skipUnless(os.environ.get('DUVERA_FULL_PROJECT'),'Supply the verified real full-film project')
    def test_actual_full_film_and_original_sound_protection(self):
        root=Path(os.environ['DUVERA_FULL_PROJECT'])
        report=json.loads((root/'production'/'delivery-qc.json').read_text(encoding='utf-8'))
        self.assertTrue(report['passed'])
        self.assertEqual(report['camera_parts'],19)
        self.assertEqual(len(report['original_sound_inserts']),4)
        self.assertTrue(all(w['original_samples_preserved_before_master'] and w['added_music_zero'] and
                            w['added_effects_zero'] for w in report['original_sound_inserts']))
        self.assertTrue(report['closing_voice_only']['music_zero'])
        self.assertTrue(report['closing_voice_only']['sfx_zero'])
        self.assertTrue(report['mastering']['final_sfx_audibility']['passed'])
        self.assertTrue(report['mastering']['visual_qc']['frame_rate_preserved'])
        self.assertTrue(report['source_receipt']['picture_clock_normalization']['encoded_packets_identical'])
        self.assertTrue(report['ebur128']['passed'])
        self.assertFalse(report['synthetic_media_used'])
