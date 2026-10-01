import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from src.organize import organize
from src.cluster import cluster_vectors

class BaselineTests(unittest.TestCase):
    def test_photo_is_copied_to_both_people_and_only_once_per_folder(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / "source"; source.mkdir()
            photo = source / "two_people.jpg"; Image.new("RGB", (40, 40)).save(photo)
            faces = pd.DataFrame({"face_id": ["a", "b", "c"], "image_path": [str(photo)] * 3})
            labels = pd.DataFrame({"face_id": ["a", "b", "c"], "cluster_id": [0, 1, 0]})
            _, summary = organize(faces, labels, source, root / "sorted")
            self.assertEqual(summary["copies"], 2)
            for folder in ["Person_01", "Person_02"]:
                self.assertEqual((root / "sorted" / folder / photo.name).read_bytes(), photo.read_bytes())

    def test_collision_zero_faces_and_filtered_faces(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / "source"; source.mkdir()
            paths = []
            for name in ["a/photo.jpg", "b/photo.jpg", "empty.jpg", "filtered.jpg"]:
                path = source / name; path.parent.mkdir(exist_ok=True)
                Image.new("RGB", (40, 40)).save(path); paths.append(str(path))
            faces = pd.DataFrame({"face_id": ["a", "b"], "image_path": paths[:2]})
            labels = pd.DataFrame({"face_id": ["a", "b"], "cluster_id": [0, 0]})
            images = pd.DataFrame({"image_path": paths, "status": ["ok"]*4, "detected": [1, 1, 0, 1]})
            _, summary = organize(faces, labels, source, root / "sorted", images)
            self.assertEqual(len(list((root / "sorted/Person_01").glob("*.jpg"))), 2)
            self.assertEqual(summary["no_face"], 1); self.assertEqual(summary["unknown"], 1)

    def test_empty_and_single_embeddings(self):
        for x in [np.empty((0, 4)), np.ones((1, 4))]:
            labels, rescued = cluster_vectors(x)
            self.assertTrue((labels == -1).all()); self.assertFalse(rescued.any())

    def test_known_groups_with_dbscan(self):
        x = np.array([[1., 0], [1., .01], [0., 1], [.01, 1], [-1., 0]])
        labels, _ = cluster_vectors(x, algorithm="dbscan", eps=.01, rescue_sim=None)
        self.assertEqual(labels[0], labels[1]); self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2]); self.assertEqual(labels[4], -1)

if __name__ == "__main__": unittest.main()
