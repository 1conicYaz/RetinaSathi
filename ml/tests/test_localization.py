from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from ml.datasets.localization import coordinates
from ml.models import DiscFoveaLocalizer,localization_loss


class LocalizationTests(unittest.TestCase):
    def test_shared_partition_csv_skips_fully_blank_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "coordinates.csv"
            path.write_text(
                "Image No,X- Coordinate,Y - Coordinate,\n"
                "train_01,12.5,42.0,\n"
                "test_01,,,\n"
            )
            self.assertEqual(coordinates(path), {"train_01": (12.5, 42.0)})

    def test_partial_coordinate_row_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "coordinates.csv"
            path.write_text(
                "Image No,X- Coordinate,Y - Coordinate\n"
                "broken,12.5,\n"
            )
            with self.assertRaisesRegex(ValueError, "Partial localization row"):
                coordinates(path)

    def test_coordinates_uncertainty_and_loss_are_valid(self)->None:
        model=DiscFoveaLocalizer(pretrained=False).eval();image=torch.randn(2,3,64,64);target=torch.rand(2,4)
        with torch.no_grad():coordinates,log_variance=model(image);loss=localization_loss(coordinates,log_variance,target)
        self.assertEqual(tuple(coordinates.shape),(2,4));self.assertEqual(tuple(log_variance.shape),(2,4))
        self.assertTrue(bool(((coordinates>=0)&(coordinates<=1)).all()));self.assertTrue(bool(torch.isfinite(loss)))


if __name__=="__main__":unittest.main()
