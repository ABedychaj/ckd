from unittest import TestCase

import pandas as pd

from app.preprocessing import encode_categorical, scale_numerical


# TODO: Add tests more tests with edge cases (e.g. empty dataframes, dataframes with only categorical or numerical columns, etc.)
# TODO: Add tests for the other functions in preprocessing.py

class TestPreprocessing(TestCase):
    def test_encode_categorical(self):
        test_df = pd.read_csv("test_data/test_data.csv")
        encoded_to_assert = pd.read_csv("test_data/test_data_encoded.csv")
        encoded = encode_categorical(test_df)
        pd.testing.assert_frame_equal(encoded, encoded_to_assert, check_dtype=False)

    def test_scale_numerical(self):
        test_df = pd.read_csv("test_data/test_data.csv")
        scaled_to_assert = pd.read_csv("test_data/test_data_scaled.csv")
        scaled = scale_numerical(test_df)
        pd.testing.assert_frame_equal(scaled, scaled_to_assert, check_dtype=False)
