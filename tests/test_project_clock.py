"""Integer-frame multi-part pause/resume contracts; no synthetic audiovisual fixtures."""
import unittest

from src.production_project import remap_part_time


class AssemblyClock(unittest.TestCase):
    def test_pause_resume_shift_applies_at_exact_insert_boundary(self):
        segments=[{'kind':'camera','part_id':'one','source_start_frame':0,'source_end_frame':600,'output_start_frame':0},
                  {'kind':'archive','frames':120,'output_start_frame':600},
                  {'kind':'camera','part_id':'one','source_start_frame':600,'source_end_frame':1200,'output_start_frame':720},
                  {'kind':'camera','part_id':'two','source_start_frame':0,'source_end_frame':600,'output_start_frame':1320}]
        self.assertEqual(remap_part_time('one',9,segments),9)
        self.assertEqual(remap_part_time('one',10,segments),12)
        self.assertEqual(remap_part_time('one',20,segments),22)
        self.assertEqual(remap_part_time('two',0,segments),22)
