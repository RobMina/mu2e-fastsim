"""Source-static contract checks only: no art import, compiler or runtime parity claim."""
from pathlib import Path
import re
import unittest


ART = Path(__file__).resolve().parents[1]
REPO = ART / 'R7Trigger'
SOURCE = (REPO / 'src/KalSeedSelectionFeatures_module.cc').read_text()


class ExporterSourceContract(unittest.TestCase):
    def test_validated_exact_tags(self):
        self.assertIn('fhicl::Atom<art::InputTag> kalSeeds', SOURCE)
        self.assertIn('fhicl::Sequence<art::InputTag> selectionInfo', SOURCE)
        for text in (
            'art::InputTag("TrigKSFDe::TriggerReplay")',
            'art::InputTag("TprTrkDe80m70pD0200KSFilter::TriggerReplay")',
            'art::InputTag("TprTrkDe80m70pKSFilter::TriggerReplay")',
            'kalSeeds_.label() != "TrigKSFDe"',
            '!kalSeeds_.instance().empty()',
            'kalSeeds_.process() != "TriggerReplay"',
            'selectionInfo_.size() != 2',
            '!tag.instance().empty()',
            'tag.process() != "TriggerReplay"',
            'expected.erase(tag.label()) != 1',
        ):
            self.assertIn(text, SOURCE)

    def test_branch_contract(self):
        branches = set(re.findall(r'events_->Branch\("([^"]+)"', SOURCE))
        expected = {
            'run', 'subrun', 'event', 'source_ordinal', 'product_present', 'product_id',
            'track_index', 'raw_status', 'valid_track', 'status_has_SeedOK',
            'has_t0_segment', 'supported_fit', 'fit_particle', 'active_hits',
            'pz', 'momentum', 'tan_dip', 'd0', 'momentum_error', 'chi2_dof',
            'fit_consistency', 't0_segment_index', 't0', 'info_present_D0200',
            'info_present_nominal', 'accepted_track_keys_D0200',
            'accepted_track_keys_nominal',
        }
        self.assertEqual(branches, expected)
        self.assertIn('std::vector<std::uint64_t> trackIndex_', SOURCE)
        self.assertIn('std::vector<std::uint32_t> rawStatus_', SOURCE)
        self.assertIn('std::vector<unsigned char> validTrack_', SOURCE)
        self.assertIn('std::vector<std::int64_t> t0SegmentIndex_', SOURCE)

    def test_exact_released_expressions_and_gates(self):
        analyze = SOURCE[SOURCE.index('void KalSeedSelectionFeatures::analyze'):]
        self.assertLess(analyze.index('if (!seedOK) continue;'), analyze.index('Ks.t0Segment(t0)'))
        for text in (
            'Ks.status().hasAllProperties(seedOK_)',
            'auto const t0seg = Ks.t0Segment(t0);',
            'if (t0seg == Ks.segments().end()) continue;',
            't0seg->momentum3()', 't0seg->position3()',
            'double const td = 1.0/tan(momvec.Theta());',
            'double const mom = momvec.R();',
            'std::copysign(lhtraj.minAxisDist(), posvec.Cross(momvec).Z())',
            'd0 = chtraj.d0();', 'd0 = kltraj.d0();',
            'activeHits_.push_back(Ks.nHits(true));',
            'momentumError_.back() = t0seg->momerr();',
            'chi2Dof_.back() = Ks.chisquared()/Ks.nDOF();',
            'fitConsistency_.back() = Ks.fitConsistency();',
            'std::numeric_limits<double>::quiet_NaN()',
            'validTrack_.push_back(1);',
        ):
            self.assertIn(text, analyze)
        self.assertLess(analyze.index('if (Ks.loopHelixFit())'),
                        analyze.index('else if (Ks.centralHelixFit())'))
        self.assertLess(analyze.index('else if (Ks.centralHelixFit())'),
                        analyze.index('else if (Ks.kinematicLineFit())'))
        self.assertNotIn('catch', analyze)
        self.assertNotIn('segments().empty()', analyze)
        self.assertNotIn('intersections(', analyze)

    def test_presence_identity_and_unconditional_fill(self):
        for text in (
            'event.getHandle<KalSeedCollection>(kalSeeds_)',
            'event.getHandle<TriggerInfo>(selectionInfo_[i])',
            'if (!productPresent_)\n      throw',
            'ptr.isNull() || ptr.id() != seeds.id() || ptr.key() >= seeds->size()',
            'if (!seen.insert(key).second)',
            'id << seeds.id();',
            'std::stoull(Ks.status().hex(), nullptr, 16)',
            '++sourceOrdinal_;',
        ):
            self.assertIn(text, SOURCE)
        self.assertEqual(SOURCE.count('events_->Fill();'), 1)
        self.assertRegex(SOURCE, r'\n  }\n  events_->Fill\(\);\n}')

    def test_no_truth_cut_emulator_or_dictionary(self):
        for text in ('MCDataProducts', 'SimParticle', 'getValidHandle', 'minMomentum',
                     'maxChi2DOF', 'minNTrks', 'gallery', 'ACLiC'):
            self.assertNotIn(text, SOURCE)
        self.assertFalse(list(REPO.rglob('classes_def.xml')))
        comments = re.findall(r'^\s*//.*$', SOURCE, re.MULTILINE)
        self.assertTrue(comments)
        self.assertNotIn('/*', SOURCE)

    def test_shared_resource_direct_link_dependency(self):
        cmake = (REPO / 'CMakeLists.txt').read_text()
        scons = (REPO / 'src/SConscript').read_text()
        libraries = cmake.split('LIBRARIES REG', 1)[1].split(')', 1)[0].split()
        self.assertIn('art::Utilities', libraries)
        self.assertRegex(scons, r"(?m)^    'art_Utilities',$")

    def test_module_registration_direct_link_dependency(self):
        cmake = (REPO / 'CMakeLists.txt').read_text()
        scons = (REPO / 'src/SConscript').read_text()
        libraries = cmake.split('LIBRARIES REG', 1)[1].split(')', 1)[0].split()
        self.assertIn('DEFINE_ART_MODULE(mu2e::KalSeedSelectionFeatures)', SOURCE)
        self.assertIn('cetlib::cetlib', libraries)
        self.assertRegex(scons, r"(?m)^    'cetlib',$")

    def test_build_library_lists_match(self):
        cmake = (REPO / 'CMakeLists.txt').read_text()
        scons = (REPO / 'src/SConscript').read_text()
        libraries = cmake.split('LIBRARIES REG', 1)[1].split(')', 1)[0].split()
        special = {'Offline': 'mu2e', 'ROOT': '', 'fhiclcpp::types': 'fhiclcpp_types',
                   'art_root_io::TFileService_service': 'art_root_io_TFileService',
                   'canvas::canvas': 'canvas', 'fhiclcpp::fhiclcpp': 'fhiclcpp',
                   'cetlib::cetlib': 'cetlib',
                   'cetlib_except::cetlib_except': 'cetlib_except'}
        translated = set()
        for library in libraries:
            if library in special:
                translated.add(special[library])
            else:
                namespace, name = library.split('::')
                prefix = special.get(namespace, namespace)
                translated.add(f'{prefix}_{name}' if prefix else name)
        actual = set(re.findall(r"^    '([^']+)',", scons, re.MULTILINE))
        self.assertEqual(translated, actual)


if __name__ == '__main__':
    unittest.main()
