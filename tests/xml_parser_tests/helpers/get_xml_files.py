import os
import random

# The following files raise parse errors but are invalid according to Musescore
INVALID_FILES = (
    "OpenScore-LiederCorpus/Elgar,_Edward/_/Is_She_Not_Passing_Fair/score.mxl",
    # Mozart K570/3 has a missing forward repeat causing Musescore to give a
    #   different result from my code; I prefer my implementation.
    "Piano_Sonatas/Mozart,_Wolfgang_Amadeus/K570/3/score.mxl",
    # Musescore prints a warning "this is not a valid file" on opening the
    # Chaminade Fleur Jetée song
    "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Chaminade,_Cécile/_/Fleur_jetée/score.mxl",
    # For an unknown reason musescore fails on this file, simply stopping
    #   100s of bars early
    "/Users/malcolm/datasets/When-in-Rome/Corpus/Quartets/Beethoven,_Ludwig_van/Op135/2/score.mxl",
)

FILES_WITH_IRREGULAR_TIMES = (
    "OpenScore-LiederCorpus/Mahler,_Gustav/Lieder_eines_fahrenden_Gesellen/1_Wenn_mein_Schatz_Hochzeit_macht/score.mxl",
    "OpenScore-LiederCorpus/Corder,_Frederick/_/O_Sun,_That_Wakenest/score.mxl",
    # OpenScore-LiederCorpus/Jaëll,_Marie/Les_Orientales/6_Vœu/score.mxl in
    #   m. 111 has 2/4 in vocal part and 6/8 in piano part. The 8ths in
    #   in the vocal part should thus be equivalent to dotted 8ths in the
    #   piano part but they have the same duration (12 ticks) as the 8ths
    #   in the piano part. Musescore is smart enough to add an extra quarter-
    #   note rest to the end of the bar to compensate
    "OpenScore-LiederCorpus/Jaëll,_Marie/Les_Orientales/6_Vœu/score.mxl",
    # OpenScore-LiederCorpus/Debussy,_Claude/Cinq_Poëmes_de_Baudelaire/3_Le_Jet_d’Eau/score.mxl
    #   seems to have a similar issue to the Jaell score above, with voice part
    #   in 3/4 and piano part in 9/8
    "OpenScore-LiederCorpus/Debussy,_Claude/Cinq_Poëmes_de_Baudelaire/3_Le_Jet_d’Eau/score.mxl",
    # 2/4 and 6/8
    "OpenScore-LiederCorpus/German,_Edward/3_Spring_Songs/1_All_the_World_Awakes_Today/score.mxl",
    # 3/2 and 9/4
    "/Users/malcolm/datasets/When-in-Rome/Corpus/OpenScore-LiederCorpus/Faltis,_Evelyn/Lieder_fernen_Gedenkens,_Op._posth/4_Heimkehr/score.mxl",
    # For an unknown reason MuseScore adds an extra dotted 8th rest to m. 8 of
    #   this Webern score
    "OpenScore-LiederCorpus/Webern,_Anton/5_Lieder_aus_“Der_siebente_Ring”,_Op.3/1_Dies_ist_ein_Lied_für_dich_allein/score.mxl",
)


def get_xml_files(
    shuffle=False, n=None, exclude_invalid_files=True, exclude_irregular_ts=True
):
    when_in_rome_dir = os.environ["WHEN_IN_ROME_DIR"]
    invalid_files = {os.path.join(when_in_rome_dir, p) for p in INVALID_FILES}
    irregular_times = {
        os.path.join(when_in_rome_dir, p) for p in FILES_WITH_IRREGULAR_TIMES
    }
    xml_files = []
    for dirname, _, filepaths in os.walk(when_in_rome_dir):
        for path in filepaths:
            if not ((path.endswith(".xml") or (path.endswith(".mxl")))):
                continue
            if path.startswith("analysis_on"):
                continue
            fullpath = os.path.join(dirname, path)
            if exclude_invalid_files and fullpath in invalid_files:
                continue
            if exclude_irregular_ts and fullpath in irregular_times:
                continue
            xml_files.append(fullpath)
    if shuffle:
        random.shuffle(xml_files)
    if n is not None:
        return xml_files[:n]
    return xml_files
