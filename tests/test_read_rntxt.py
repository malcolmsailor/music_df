import os
import tempfile
from io import StringIO

import pandas as pd

from music_df.read_rntxt import read_rntxt

KERN_CONTENTS = """Composer: J. S. Bach
BWV: 153.1
Title: Ach Gott, vom Himmel sieh' darein
Analyst: Andrew Jones
Proofreader: Dmitri Tymoczko and Hamish Robb
Note: please email corrections to dmitri@princeton.edu

Time Signature: 4/4

Form: chorale
Note: piece has decidedly minor-mixolydian feel
m0 b4 a: V
m1 i b2 viio6 b3 i6 b4 V4/3 b4.5 i
m2 V6 b1.5 V6/5 b2 i b3 V || b4 viio6/5
m3 i6 b2 V b2.5 V7 b3 VI b4 iio6
m4 i6/4 b2 V b3 i || b4 G: V6
m5 I b1.5 e: viio6 b2 i b3 V b3.5 V2 b4 i6 b4.5 viio6
Note: parallel fifths evaded by voice crossing in m. 6
Time Signature: 4/2
m6 i b2 iv6 b3 V || b4 i
m6var1 i b2 iio6/4 b2.5 ii/o4/3 b3 V || b4 i
m7 a: VI b2 i6 b3 V b4 i
m8 i6 b2 V b3 i || b4 i
m9 V6 b2 i b3 iv6 b4 viio7/IV
m10 IV b1.5 viio7/V b3 V
"""


def test_read_rntxt():
    # write KERN_CONTENTS to temporary file
    _, temp_rntxt_path = tempfile.mkstemp(suffix=".txt")
    with open(temp_rntxt_path, "w") as outf:
        outf.write(KERN_CONTENTS)
    # read the temporary file
    df = read_rntxt(temp_rntxt_path)
    os.remove(temp_rntxt_path)

    # expected_df = pd.read_csv(StringIO(EXPECTED_RESULT), quotechar="'")
    # assert df.equals(expected_df)
