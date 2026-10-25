"""V2 regressions use timestamp contracts and real supplied media, no dummy AV demos."""
import json
import os
import unittest
import tempfile
import shutil
from pathlib import Path

from pydantic import ValidationError

from src.documentary_schema import QuoteLine, VisualEditEvent
from src.documentary_pipeline import build_time_map, effects_and_silence
from src.documentary_schema import DocumentaryEditPlan
from src.models import Segment, Transcript, Word
from src.retake_cutter import detect_retakes, splice_switches


def transcript(tokens):
    words=[Word(word=t,start=a,end=b) for t,a,b in tokens]
    return Transcript(timing_source='asr',words=words,segments=[Segment(start=0,end=10,text='source words')])


class RetakeContracts(unittest.TestCase):
    def test_fuzzy_restart_removes_only_first_take_and_clock_switches_on_splice(self):
        raw=transcript([('क्योंकि',1,1.3),('इस',1.3,1.5),('प्रोटेस्ट',1.5,1.9),('में',1.9,2.1),
                        ('क्योंकि',2.4,2.7),('इस',2.7,2.9),('प्रोटेस्ट',2.9,3.2),('ने',3.2,3.4)])
        cuts,_=detect_retakes(raw,None)
        self.assertEqual(len(cuts),1)
        self.assertEqual(cuts[0]['start_sec'],1)
        self.assertEqual(cuts[0]['end_sec'],2.4)
        mapping,_=build_time_map(DocumentaryEditPlan(),raw,None,[],{'rate':'60','frames':600},cuts)
        self.assertEqual(mapping.time(2.4),1)
        self.assertEqual(splice_switches(mapping)[0]['frame'],60)
        self.assertEqual(splice_switches(mapping)[0]['scale'],1.18)

    def test_shared_prefix_in_distinct_complete_sentences_is_preserved(self):
        tokens=['लड़कों','ने','इसका','बेस्ट','रिस्पांस','दिया','ऑफेंड','होने','के','बजाय','लड़कों','ने','वाइब','किया']
        raw=transcript([(t,.2*i,.2*i+.15) for i,t in enumerate(tokens)])
        cuts,_=detect_retakes(raw,None)
        self.assertEqual(cuts,[])

    def test_repeated_phonetic_variant_matches_above_point75(self):
        raw=transcript([('jawab',1,1.3),('mein',1.3,1.5),('reels',1.5,1.9),
                        ('jawab',2.4,2.7),('mein',2.7,2.9),('reel',2.9,3.2)])
        self.assertEqual(len(detect_retakes(raw,None)[0]),1)

    def test_rhyming_slogan_words_are_not_deleted_as_fuzzy_retakes(self):
        raw=transcript([('दर्द',1,1.3),('है',1.3,1.5),('जिसका',1.5,1.9),('कारण',1.9,2.1),
                        ('मर्द',2.4,2.7),('है',2.7,2.9)])
        self.assertEqual(detect_retakes(raw,None)[0],[])

    def test_three_token_rhyming_slogan_preserved(self):
        raw=transcript([('है',1,1.2),('मर्दों',1.2,1.6),('में',1.6,1.8),('औरत',1.8,2),('क्यों',2,2.2),
                        ('है',2.3,2.5),('पर्दों',2.5,2.8),('में',2.8,3)])
        self.assertEqual(detect_retakes(raw,None)[0],[])

    def test_longest_restart_preserves_preceding_sentence_suffix(self):
        raw=transcript([('घूम',1,1.2),('रही',1.2,1.4),('थी',1.4,1.6),('कहा',1.7,1.9),('जा',1.9,2.1),
                        ('रही',2.1,2.3),('थी',2.3,2.5),('कि',2.5,2.7),('कहा',3,3.2),('जा',3.2,3.4),
                        ('रही',3.4,3.6),('थी',3.6,3.8),('कि',3.8,4),('लड़के',4,4.3)])
        cuts,_=detect_retakes(raw,None)
        self.assertEqual(len(cuts),1)
        self.assertGreater(cuts[0]['start_sec'],1.6)
        self.assertEqual(cuts[0]['end_sec'],3)


class EditorialContracts(unittest.TestCase):
    def test_unrelated_publisher_headline_cannot_label_a_narration_card(self):
        from src.evidence_fetcher import verify_headline_excerpt
        from src.media import PipelineError
        verify_headline_excerpt('Delhi protest slogans','Latest: Delhi protest slogans | News')
        with self.assertRaisesRegex(PipelineError,'not an excerpt'):
            verify_headline_excerpt('Offend hone ke bajaye vibe karna','Delhi protest slogans')

    def test_narration_labels_rejected(self):
        for label in ('NARRATION // EXCERPT','SPOKEN IN THE SOURCE','OPINION & COMMENTARY'):
            with self.assertRaises(ValidationError):
                VisualEditEvent(start_sec=0,end_sec=10,edit_type='evidence_card',headline_text='Example',metadata_label=label)

    def test_complete_slogans_need_every_word_timestamp(self):
        with self.assertRaises(ValidationError):
            VisualEditEvent(start_sec=0,end_sec=8,edit_type='slogan_cards',headline_text='Slogans',
                            quote_lines=[QuoteLine(text='one two three',start_sec=.5,word_times=[.5,1])])

    def test_every_card_has_required_layer_bundle_and_marker_delay(self):
        events=[VisualEditEvent(event_id='news',start_sec=0,end_sec=8,edit_type='evidence_card',headline_text='A real news headline',required=True),
                VisualEditEvent(event_id='chapter',start_sec=10,end_sec=18,edit_type='black_chapter_card',headline_text='Why follow?',required=True),
                VisualEditEvent(event_id='cite',start_sec=20,end_sec=25,edit_type='lower_third_citation',headline_text='A real source',required=True)]
        graph,silence=effects_and_silence(events)
        self.assertEqual(len(graph.events),5)
        marker=next(e for e in graph.events if e.event_id=='edl_news_layer1')
        self.assertEqual(marker.time_sec,.3)
        self.assertEqual(marker.target_peak_dbfs,-8)
        self.assertEqual(len(silence[0].accent_event_ids),2)


class RealV2Delivery(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('DUVERA_REAL_V2'),'Supply the real V2 render to verify delivery')
    def test_encoded_program_and_retakes(self):
        root=Path(os.environ['DUVERA_REAL_V2'])
        report=json.loads((root/'delivery-qc.json').read_text(encoding='utf-8'))
        self.assertTrue(report['passed'])
        self.assertEqual(report['retake_count'],2)
        self.assertLess(report['output_duration_sec'],50)
        self.assertTrue(report['final_sfx_audibility']['passed'])
        self.assertTrue(report['thesis_music_silence']['passed'])
        self.assertFalse(report['synthetic_av_used'])

    @unittest.skipUnless(os.environ.get('DUVERA_REAL_V2'),'Supply real recorded/encoded audio for the missing-SFX negative gate')
    def test_gate_rejects_actual_encoded_counterfactual_with_no_sfx(self):
        from src.documentary_audio import verify_final_sfx_audibility
        from src.media import PipelineError
        from src.pacing_models import PacingPlan
        root=Path(os.environ['DUVERA_REAL_V2'])
        render=root/'render-final'
        branch=render/'elevenlabs'
        pacing=PacingPlan.model_validate_json((render/'pacing_plan.json').read_text())
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            check=Path(temporary)
            for name in ('encoded_without_sfx.m4a','audibility_music_upper_bound.wav'):
                shutil.copyfile(branch/name,check/name)
            with self.assertRaisesRegex(PipelineError,'audibility gate failed'):
                verify_final_sfx_audibility(check/'encoded_without_sfx.m4a',check,pacing)
