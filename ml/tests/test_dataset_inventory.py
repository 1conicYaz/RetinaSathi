from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "inventory_datasets.py"
SPEC = importlib.util.spec_from_file_location("inventory_datasets", SCRIPT)
assert SPEC and SPEC.loader
inventory_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = inventory_module
SPEC.loader.exec_module(inventory_module)


class DatasetInventoryTests(unittest.TestCase):
    def test_paths_are_private_and_roles_are_separated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            aptos = root / "APTOS_2019"
            (aptos / "extracted/train_images").mkdir(parents=True)
            (aptos / "raw").mkdir()
            (aptos / "extracted/train_images/example.png").write_bytes(b"image")
            with (aptos / "extracted/train.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["id_code", "diagnosis"])
                writer.writerow(["example", 2])
            (aptos / "raw/source.zip").write_bytes(b"archive")

            result = inventory_module.build_inventory(root)
            dataset = result["datasets"][0]
            self.assertEqual(result["dataset_root"], "${RETINASATHI_DATA_ROOT}")
            self.assertEqual(dataset["counts"]["image"], 1)
            self.assertEqual(dataset["counts"]["annotation"], 1)
            self.assertEqual(dataset["counts"]["archive"], 1)
            self.assertEqual(dataset["annotations"][0]["rows"], 1)
            self.assertNotIn(temporary, str(result))

    def test_idrid_official_test_policy_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = inventory_module.build_inventory(Path(temporary))
            idrid = next(item for item in result["datasets"] if item["name"] == "IDRiD")
            self.assertIn("official test is locked", idrid["split_policy"].lower())


if __name__ == "__main__":
    unittest.main()
