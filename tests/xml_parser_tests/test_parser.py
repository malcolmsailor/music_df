import math
import os
import random
import subprocess

import numpy as np
import pandas as pd
import pytest

from music_df.midi_parser import midi_to_table
from music_df.xml_parser import xml_parse
from tests.xml_parser_tests.helpers.get_xml_files import get_xml_files
from tests.xml_parser_tests.helpers.mscore_midi_conversion import musescore_to_midi
from tests.xml_parser_tests.helpers.run_scorext import run_scorext

MODULE_DIR = os.path.dirname((os.path.realpath(__file__)))


def test_parse():
    xml_files = [
        os.path.join(MODULE_DIR, "resources", p)
        for p in os.listdir(os.path.join(MODULE_DIR, "resources"))
        if (p.endswith(".xml") or p.endswith(".mxl") or p.endswith(".mscx"))
    ]
    for xml_file in xml_files:
        df = xml_parse(xml_file)


def test_expand_repeats():
    # There is a dacapo in n13op130_06 which defeats my repeat parsing. Not
    #   sure if it is plausibly parseable.
    # df = parse(
    #     "/Users/malcolm/google_drive/python/third_party/ABC/MS3/n13op130_06.mscx",
    #     expand_repeats="drop",
    # )

    def _run_assertions(df, onsets_and_releases, pitches):
        onsets, releases = zip(*onsets_and_releases)
        assert (df[df.type == "note"].pitch == pitches).all()
        assert (df[df.type == "note"].onset == list(onsets)).all()
        assert (df[df.type == "note"].release == list(releases)).all()
        assert (df[df.type == "bar"].onset == list(onsets)).all()
        assert (df[df.type == "bar"].release == list(releases)).all()

    path = os.path.join(MODULE_DIR, "resources", "repeats.xml")

    # drop
    onsets_and_releases = [
        (0, 4),
        (4, 8),
        (8, 12),
        (12, 16),
        (16, 19),
        (19, 22),
    ]
    pitches = [72, 71, 67, 65, 64, 59]
    df = xml_parse(path, expand_repeats="drop")
    _run_assertions(df, onsets_and_releases, pitches)

    # yes
    onsets_and_releases = [
        (0, 4),
        (4, 8),
        (8, 12),
        (12, 16),
        (16, 20),
        (20, 24),
        (24, 28),
        (28, 31),
        (31, 34),
        (34, 37),
        (37, 40),
        (40, 43),
        (43, 46),
    ]
    pitches = [72, 72, 71, 69, 71, 67, 65, 64, 62, 64, 60, 64, 59]
    df = xml_parse(path, expand_repeats="yes")
    _run_assertions(df, onsets_and_releases, pitches)

    # no
    onsets_and_releases = (
        (0, 4),
        (4, 8),
        (8, 12),
        (12, 16),
        (16, 20),
        (20, 23),
        (23, 26),
        (26, 29),
        (29, 32),
    )
    pitches = [72, 71, 69, 67, 65, 64, 62, 60, 59]
    df = xml_parse(path, expand_repeats="no")
    _run_assertions(df, onsets_and_releases, pitches)


@pytest.mark.skip(
    reason="""I think these files have various errors (e.g., bars where 
    different parts aren't the same length). To explore further..."""
)
def test_wir():
    xml_files = get_xml_files()
    failed = []
    xml_files = [
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Mahler,_Gustav/Lieder_eines_fahrenden_Gesellen/4_Die_zwei_blauen_Augen_von_meinem_Schatz/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Mahler,_Gustav/Lieder_eines_fahrenden_Gesellen/3_Ich_hab’_ein_glühend_Messer/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Brahms,_Johannes/2_Gesänge,_Op.91/2_Geistliches_Wiegenlied/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Brahms,_Johannes/2_Gesänge,_Op.91/1_Gestillte_Sehnsucht/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Chaminade,_Cécile/_/L’absente/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Boulanger,_Lili/Clairières_dans_le_ciel/03_Parfois,_je_suis_triste/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Boulanger,_Lili/_/Reflets/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Boulton,_Harold/12_New_Songs/07_A_Song_of_the_South/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Holmès,_Augusta_Mary_Anne/Les_Sept_Ivresses/7_L’Or/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Holmès,_Augusta_Mary_Anne/Contes_divins/2_Les_lys_bleus/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Holmès,_Augusta_Mary_Anne/Les_Heures/1_L’Heure_Rose/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Holmès,_Augusta_Mary_Anne/Mélodies_pour_piano_et_chant/19_Un_rêve, à_2_voix/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Gambarini,_Elisabetta_de/_/Behold,_Behold_and_Listen/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Debussy,_Claude/Cinq_Poëmes_de_Baudelaire/4_Recueillement/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Debussy,_Claude/Cinq_Poëmes_de_Baudelaire/3_Le_Jet_d’Eau/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Schumann,_Robert/Liederkreis,_Op.39/05_Mondnacht/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Schumann,_Robert/Frauenliebe_und_Leben,_Op.42/3_Ich_kann’s_nicht_fassen/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Munktell,_Helena/10_Songs/02_Dans_le_lointain_des_bois/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Munktell,_Helena/10_Songs/01_Sérénade/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Fauré,_Gabriel/Cinq_Mélodies,_Op.58/2_En_sourdine/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Chabrier,_Emmanuel/_/Chanson_pour_Jeanne/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Strauss,_Richard/4_Lieder,_Op.27/3_Heimliche_Aufforderung/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Cornelius,_Peter/An_Bertha,_Op.15/1_Sei_mein!/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Warlock,_Peter/Songs/13_Lullaby/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Warlock,_Peter/Songs/08_Walking_the_woods/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Faltis,_Evelyn/Lieder_fernen_Gedenkens,_Op._posth/3_Sprich/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Faltis,_Evelyn/Lieder_fernen_Gedenkens,_Op._posth/4_Heimkehr/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Reichardt,_Louise/Zwölf_Gesänge/12_Aus_Novalis_Hymnen_an_die_Nacht/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Jaëll,_Marie/4_Mélodies/3_Les_petits_oiseaux/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Jaëll,_Marie/Les_Orientales/6_Vœu/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Jaëll,_Marie/Les_Orientales/2_Nourmahal-la-Rousse/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Jaëll,_Marie/Les_Orientales/3_Clair_de_lune/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Jaëll,_Marie/Les_Orientales/4_Les_tronçons_du_serpent/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Jaëll,_Marie/La_mer/2_Causeries_de_vagues/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Lehmann,_Liza/5_Little_Love_Songs/4_If_I_were_a_bird,_I_would_sing_all_day/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Lehmann,_Liza/Songs_of_Love_and_Spring/02_When_Young_Love_Comes_Knocking/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/German,_Edward/3_Spring_Songs/1_All_the_World_Awakes_Today/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Schubert,_Franz/Schwanengesang,_D.957/11_Die_Stadt/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Wolf,_Hugo/Eichendorff-Lieder/08_Nachtzauber/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Viardot,_Pauline/_/Le_chêne_et_le_roseau,_VWV_1036/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Faisst,_Clara_Mathilda/4_Lieder,_Op._11/4_Dulde,_gedulde_dich_fein/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Faisst,_Clara_Mathilda/5_Lieder,_Op.6/1_Ich_glaub’,_lieber_Schatz/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Webern,_Anton/5_Lieder_aus_“Der_siebente_Ring”,_Op.3/2_Im_Windesweben/score.mxl",
    ]
    for i, xml_file in enumerate(xml_files):
        # if "The_Shepherd’s_Song" not in xml_file:
        #     continue
        # if (
        #     "/Users/malcolm/datasets/When-in-Rome/Corpus/Quartets/Beethoven,_Ludwig_van/Op059_No1/1/score.mxl"
        #     not in xml_file
        # ):
        #     continue
        print(f"{i + 1}/{len(xml_files)}: {xml_file}")
        # df = parse(xml_file)
        try:
            df = xml_parse(xml_file)
        except:
            print("FAILED")
            failed.append(xml_file)
    print(f"FAILED ON {len(failed)} FILES")
    for f in failed:
        print(f)


def _process_df(df: pd.DataFrame):
    df = df.loc[:, ["pitch", "onset", "release"]]
    # df.sort_values(by="release", inplace=True, ignore_index=True)
    # df.sort_values(
    #     by="pitch", inplace=True, ignore_index=True, kind="mergesort"
    # )
    # df.sort_values(
    #     by="onset", inplace=True, ignore_index=True, kind="mergesort"
    # )
    return df


@pytest.mark.skip(reason="musicxml2hum/scorext don't give reliable results")
def test_compare_parses_with_scorext():
    xml_files = get_xml_files()
    for i, xml_file in enumerate(xml_files):
        print(f"{i + 1}/{len(xml_files)}: {xml_file}")
        try:
            scorext_result = run_scorext(xml_file)
        except subprocess.CalledProcessError:
            print("musicxml2hum parse error")
            continue
        parse_result = xml_parse(xml_file)

        scorext_result = _process_df(scorext_result)
        parse_result = _process_df(parse_result)


@pytest.mark.skip(reason="this test needs to be revised, I believe")
def test_compare_parses_with_mscore():
    passing_different_final_onsets = [
        # "When-in-Rome/Corpus/OpenScore-LiederCorpus/Brahms,_Johannes/7_Lieder,_Op.48/3_Liebesklage_des_Mädchens/score.mxl",
        # Hensel op 10/5 ends with arpeggio
        # "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Hensel,_Fanny_(Mendelssohn)/5_Lieder,_Op.10/5_Bergeslust/score.mxl",
        # This Reichardt score has a (1, 3), (2, 4) ending schema---TODO!
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Reichardt,_Louise/Zwölf_Gesänge/03_Nach_Sevilla/score.mxl",
        # Coleridge ends with arpeggio
        # "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Coleridge-Taylor,_Samuel/6_Sorrow_Songs,_Op.57/5_Unmindful_of_the_Roses/score.mxl",
        # Schubert ends with arpeggio
        # "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Schubert,_Franz/_/Das_Rosenband,_D.280/score.mxl",
        # Schubert ends with arpeggio
        # "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Schubert,_Franz/Winterreise,_D.911/07_Auf_dem_Flusse/score.mxl",
        # "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Schubert,_Franz/Winterreise,_D.911/02_Die_Wetterfahne/score.mxl",
        # "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Lang,_Josephine/6_Lieder,_Op.26/1_Wiegenlied/score.mxl",
        # "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Wolf,_Hugo/Eichendorff-Lieder/14_Der_verzweifelte_Liebhaber/score.mxl",
    ]
    da_segno_scores = [
        "When-in-Rome/Corpus/OpenScore-LiederCorpus/Hensel,_Fanny_(Mendelssohn)/6_Lieder,_Op.9/6_Die_Mainacht/score.mxl",
        # The following Reichardt score seems to have incoherent repeats---A repeat bar, a DC, AND a DS
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Reichardt,_Louise/Zwölf_Gesänge/01_Erinnrung_zum_Bach/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Reichardt,_Louise/Zwölf_Gesänge/02_Der_Sänger_geht/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Reichardt,_Louise/Zwölf_Gesänge/04_Vaters_Klage/score.mxl",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Grandval,_Clémence_de/6_Nouvelles_mélodies/4_Mignonne/score.mxl",
        "OpenScore-LiederCorpus/Gonzaga,_Chiquinha/_/Feijoada_do_Brasil/score.mxl",
    ]
    random.seed(42)
    xml_files = get_xml_files(shuffle=True, n=50)
    for i, xml_file in enumerate(xml_files):
        # if (
        #     xml_file
        #     != "/Users/malcolm/datasets/When-in-Rome/Corpus/Piano_Sonatas/Mozart,_Wolfgang_Amadeus/K570/3/score.mxl"
        # ):
        #     continue
        print(f"{i + 1}/{len(xml_files)}: {xml_file}")

        midi_path = musescore_to_midi(xml_file, only_if_newer=True)
        if not midi_path:
            print("No cached midi")
            continue
        mdf = midi_to_table(
            midi_path,
            pitch_sort_asc=True,
            track_sort_asc=None,
            notes_only=True,
            # it seems that "end_last" is the best setting;
            # what would be nice would be if music21/hum2mid would write multiple
            # voices on the same staff to different channels
            overlapping_notes="end_last",
        )
        pdf = _process_df(xml_parse(xml_file))
        passing_different_final_onset = any(
            xml_file.endswith(f) for f in passing_different_final_onsets
        )
        da_segno = any(xml_file.endswith(f) for f in da_segno_scores)
        if len(mdf) != len(pdf):
            if not da_segno:
                print("Frame lengths do not match")
                # TODO some tests are not passing
        else:
            pitches_match = (pdf.pitch == mdf.pitch).all()
            # numpy "isclose" doesn't work because of fractions
            ABS_TOLERANCE = 0.001
            onsets_match = (pdf.onset - mdf.onset).abs().max() < ABS_TOLERANCE
            print(f"Pitches match: {pitches_match}")
            print(f"Onsets match: {onsets_match}")
            if (not pitches_match) or (not onsets_match):
                breakpoint()
        assert pdf.loc[0, "onset"] == mdf.loc[0, "onset"]
        try:
            assert pdf.iloc[-1].onset == mdf.iloc[-1].onset
        except AssertionError:
            try:
                assert passing_different_final_onset or da_segno
            except AssertionError:
                assert abs(pdf.iloc[-1].onset - mdf.iloc[-1].onset) < 0.125
