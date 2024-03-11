# Deprecated in favor of doctest in function definition
# import os

# import pytest

# from music_df import read_csv
# from music_df.dedouble import dedouble
# from tests.helpers_for_tests import get_input_kern_paths

# SCRIPT_DIR = os.path.dirname((os.path.realpath(__file__)))
# DOUBLED_CSV = os.path.join(SCRIPT_DIR, "resources", "doubled.csv")


# # @pytest.mark.parametrize("kern_file", get_input_kern_paths(seed=42))
# def test_dedouble():
#     df = read_csv(DOUBLED_CSV)
#     assert df is not None
#     dedoubled_df = dedouble(df)
#     dedoubled_df = dedouble(df, columns=("type", "pitch", "track"))
