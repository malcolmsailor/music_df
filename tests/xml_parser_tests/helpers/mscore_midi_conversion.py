import argparse
import functools
import json
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
import typing as t
import xml.sax
from collections import defaultdict
from contextlib import contextmanager
from zipfile import ZipFile

import pandas as pd

from tests.xml_parser_tests.helpers.get_xml_files import get_xml_files

MIDI_CACHE = os.environ.get("MIDI_CACHE", None)


class MxlMetaHandler(xml.sax.ContentHandler):
    def __init__(self):
        super().__init__()
        self.musicxml_path: t.Optional[str] = None

    def startElement(self, name, attrs):
        if name == "rootfile" and self.musicxml_path is None:
            self.musicxml_path = attrs["full-path"]

    def endDocument(self):
        assert self.musicxml_path is not None


@contextmanager
def handle_xml(path):
    if path.endswith(".xml"):
        with open(path) as inf:
            yield inf
    elif path.endswith("mxl"):
        archive = ZipFile(path)
        handler = MxlMetaHandler()
        with archive.open("META-INF/container.xml") as inf:
            xml.sax.parse(inf, handler)
        with archive.open(handler.musicxml_path) as inf:
            yield inf
        # return handler.musicxml_path
    else:
        raise NotImplementedError


def _remove_graces(contents):
    # According to the xml standard the <grace/> (if present) should always be
    #   the first element in a <note>, so this regex should work
    # https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/note/
    return re.sub(
        r"\n?\s*<note\b[^>]*>\n?\s+<grace.*?/>.*?</note>",
        "",
        contents,
        flags=re.DOTALL + re.MULTILINE,
    )


def _remove_element_with_start_and_end_tags(contents, element_name):
    return re.sub(
        rf"<{element_name}\b.*?</{element_name}>",
        "",
        contents,
        flags=re.DOTALL + re.MULTILINE,
    )


def _remove_element_with_start_tag_only(contents, element_name):
    return re.sub(
        rf"<{element_name}\b.*?/>",
        "",
        contents,
        flags=re.DOTALL + re.MULTILINE,
    )


@contextmanager
def preprocess_musescore(
    path,
    remove_harmonies=True,
    remove_arpeggios=True,
    remove_ornaments=True,
    remove_graces=True,
):
    # This context manager is a hack to remove harmonies from
    #   xml or mxl files, because they get "played back" (i.e., added
    #   to the exported midi file) when exporting with musescore.
    #   See this issue: https://musescore.org/en/node/335103
    # xml_path = preprocess_musescore(path)
    # TODO remove grace notes!
    if not any([remove_harmonies, remove_arpeggios, remove_ornaments, remove_graces]):
        yield path
        return
    _, tmp_xml = tempfile.mkstemp(suffix=".xml")
    try:
        with handle_xml(path) as inf:
            contents = inf.read()
            if isinstance(contents, bytes):
                contents = contents.decode()
            if remove_harmonies:
                # contents = re.sub(
                #     f"<harmony.*?</harmony>",
                #     "",
                #     contents,
                #     flags=re.DOTALL + re.MULTILINE,
                # )
                contents = _remove_element_with_start_and_end_tags(contents, "harmony")
            if remove_arpeggios:
                contents = _remove_element_with_start_tag_only(contents, "arpeggiate")
                # NB according to the MusicXML standard, non-arpeggiate
                #   means *not* arpeggiate a chord, however, musescore seems
                #   to interpret it as meaning to arpeggiate.
                contents = _remove_element_with_start_tag_only(
                    contents, "non-arpeggiate"
                )
            if remove_ornaments:
                # TODO eventually I would like to preserve at least tremoli
                contents = _remove_element_with_start_and_end_tags(
                    contents, "ornaments"
                )
                contents = _remove_element_with_start_and_end_tags(contents, "slide")
                contents = _remove_element_with_start_tag_only(contents, "slide")
                contents = _remove_element_with_start_and_end_tags(
                    contents, "glissando"
                )
                contents = _remove_element_with_start_tag_only(contents, "glissando")
            if remove_graces:
                contents = _remove_graces(contents)
            with open(tmp_xml, "w") as outf:
                outf.write(contents)
            yield tmp_xml
    finally:
        os.remove(tmp_xml)


def musescore_to_midi(in_path, only_if_newer=True, only_if_cached=False) -> str:
    assert MIDI_CACHE is not None
    breakpoint()
    out_path = os.path.join(
        MIDI_CACHE,
        os.path.splitext(in_path)[0].lstrip(os.path.sep) + ".mid",
    )
    if only_if_cached and not os.path.exists(out_path):
        return ""

    if not shutil.which("mscore"):
        raise ValueError("mscore must be in path")

    if (
        only_if_newer
        and os.path.exists(out_path)
        and os.path.getmtime(out_path) > os.path.getmtime(in_path)
    ):
        # print(f"Using cached version of {out_path}")
        return out_path

    with preprocess_musescore(in_path) as xml_path:
        commands = ["mscore", f"{xml_path}", "-o", f"{out_path}"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        subprocess.run(commands, check=True, capture_output=True)
        print(f"Wrote {out_path}")

    return out_path


def filter_files(pattern, paths, max_count=None):
    out = []
    for path in paths:
        with handle_xml(path) as inf:
            contents = inf.read()
            if isinstance(contents, bytes):
                contents = contents.decode()
        if re.search(pattern, contents):
            out.append(path)
            if max_count is not None and len(out) == max_count:
                break
    return out


def test_preprocess_musescore():
    with preprocess_musescore(
        "/Users/malcolm/datasets/When-in-Rome/Corpus/Quartets/Beethoven,_Ludwig_van/Op018_No5/3/score.mxl"
    ) as xml_path:
        with open(xml_path) as inf:
            contents = inf.read()
    with open(
        os.path.join(os.path.dirname((os.path.realpath(__file__))), "temp.xml"),
        "w",
    ) as outf:
        outf.write(contents)


if __name__ == "__main__":
    ff = filter_files(re.compile(r"sound segno="), get_xml_files())
    breakpoint()
