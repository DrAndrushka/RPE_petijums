"""Focused regression tests for RPE_petijums dev pipeline correctness fixes."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from cleaning import apply_schema
from eda import _encode_binary_target
from inferential import _build_design, _encode_target, _safe_z_denominator
from report import _embed_svg_src, svg_grid
from schema_infer import ColSpec, infer_schema


class TestBinaryPositiveClass(unittest.TestCase):
  def test_encode_respects_false_positive_class(self):
    y = pd.Series([False, False, True, True], name="event")
    auto, _ = _encode_binary_target(y, None)
    explicit, _ = _encode_binary_target(y, False)
    # Auto picks True -> 0,0,1,1
    np.testing.assert_array_equal(auto.values, [0.0, 0.0, 1.0, 1.0])
    # False is positive -> 1,1,0,0
    np.testing.assert_array_equal(explicit.values, [1.0, 1.0, 0.0, 0.0])
    self.assertFalse(np.allclose(auto, explicit))

  def test_plot_rate_counts_not_inverted(self):
    """Level A: both False; level B: both True — P(event) differs by positive_class."""
    df = pd.DataFrame({"y": [False, False, True, True], "x": ["A", "A", "B", "B"]})
    mask_a = df["x"] == "A"
    mask_b = df["x"] == "B"
    auto, _ = _encode_binary_target(df["y"], None)
    pos_false, _ = _encode_binary_target(df["y"], False)
    p_auto_a = auto.loc[mask_a].mean()
    p_false_a = pos_false.loc[mask_a].mean()
    p_auto_b = auto.loc[mask_b].mean()
    p_false_b = pos_false.loc[mask_b].mean()
    self.assertEqual(p_auto_a, 0.0)
    self.assertEqual(p_false_a, 1.0)
    self.assertEqual(p_auto_b, 1.0)
    self.assertEqual(p_false_b, 0.0)
    self.assertNotEqual(p_auto_a, p_false_a)


class TestInferentialEncoding(unittest.TestCase):
  def test_missing_target_stays_nan(self):
    y = pd.Series([True, False, pd.NA], dtype="boolean")
    enc, _ = _encode_target(y, True)
    self.assertTrue(np.isnan(enc.iloc[2]))
    self.assertEqual(enc.iloc[0], 1.0)
    self.assertEqual(enc.iloc[1], 0.0)

  def test_z_denominator_nan_and_zero(self):
    self.assertEqual(_safe_z_denominator(0.0), 1.0)
    self.assertEqual(_safe_z_denominator(float("nan")), 1.0)
    self.assertEqual(_safe_z_denominator(2.5), 2.5)

  def test_dummy_reindex_fill_zero(self):
    schema = {
      "g": ColSpec("g", "nominal"),
    }
    f0 = pd.DataFrame({"g": ["a", "a", "b"]})
    f1 = pd.DataFrame({"g": ["b", "b", "b"]})  # no "a" -> dummy only for b
    X0, _ = _build_design(f0, schema, ["g"])
    keep = list(X0.columns)
    X1, _ = _build_design(f1, schema, ["g"])
    aligned = X1.reindex(columns=keep, fill_value=0.0)
    self.assertEqual(list(aligned.columns), keep)
    self.assertTrue((aligned.fillna(-99) >= 0).all().all())


class TestSchemaAndCleaning(unittest.TestCase):
  def test_unique_continuous_not_marked_id(self):
    n = 50
    df = pd.DataFrame({"measurement": np.arange(n, dtype=float)})
    schema = infer_schema(df)
    self.assertEqual(schema["measurement"].kind, "continuous")

  def test_ordinal_without_levels_raises(self):
    df = pd.DataFrame({"grade": [1, 2, 3]})
    schema = {"grade": ColSpec("grade", "ordinal")}
    with self.assertRaises(ValueError):
      apply_schema(df, schema)


class TestReportEmbed(unittest.TestCase):
  def test_svg_embedded_as_data_uri(self):
    with tempfile.TemporaryDirectory() as tmp:
      fig = Path(tmp) / "plot.svg"
      fig.write_text('<svg xmlns="http://www.w3.org/2000/svg"><circle r="1"/></svg>')
      uri = _embed_svg_src(fig)
      self.assertIsNotNone(uri)
      self.assertTrue(uri.startswith("data:image/svg+xml;base64,"))
      html = svg_grid([fig])
      self.assertIn("data:image/svg+xml;base64,", html)
      self.assertNotIn("../", html)
      self.assertNotIn("file://", html)


if __name__ == "__main__":
  unittest.main()
