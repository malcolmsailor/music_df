import os
import tempfile
from io import StringIO

import pandas as pd

from music_df.read_krn import read_krn

KERN_CONTENTS = """**kern	**kern
*Ibass	*Ioboe
*M3/4	*M3/4
8C	12d
.	12e
8B	.
.	12f
*	*^
4A	2g	4d
4G	.	4c
*	*v	*v
=	=
*-	*-
"""

EXPECTED_RESULT = r"""onset,type,track,instrument,pitch,release,spelling,other
0.000000,bar,,,,3.000000,,
0.000000,time_signature,,,,,,'{"numerator": 3, "denominator": 4}'
0.000000,note,1.0,bass,48.0,0.500000,C,
0.000000,note,2.0,oboe,62.0,0.333333,D,
0.333333,note,2.0,oboe,64.0,0.666667,E,
0.500000,note,1.0,bass,59.0,1.000000,B,
0.666667,note,2.0,oboe,65.0,1.000000,F,
1.000000,note,1.0,bass,57.0,2.000000,A,
1.000000,note,2.1,oboe,67.0,3.000000,G,
1.000000,note,2.2,oboe,62.0,2.000000,D,
2.000000,note,1.0,bass,55.0,3.000000,G,
2.000000,note,2.2,oboe,60.0,3.000000,C,"""


def test_read_krn():
    # write KERN_CONTENTS to temporary file
    _, temp_krn_path = tempfile.mkstemp(suffix=".krn")
    with open(temp_krn_path, "w") as outf:
        outf.write(KERN_CONTENTS)
    # read the temporary file
    df = read_krn(temp_krn_path)
    os.remove(temp_krn_path)

    expected_df = pd.read_csv(StringIO(EXPECTED_RESULT), quotechar="'")
    assert df.equals(expected_df)
