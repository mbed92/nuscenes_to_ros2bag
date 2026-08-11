import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from builtin_interfaces.msg import Time
from sensor_msgs.msg import PointCloud2, PointField

utils = types.ModuleType("utils")
utils.np = np
utils.PointCloud2 = PointCloud2
utils.PointField = PointField

def get_time(data):
    stamp = Time()
    stamp.sec, microseconds = divmod(data["timestamp"], 1_000_000)
    stamp.nanosec = microseconds * 1000
    return stamp

utils.get_time = get_time
sys.modules["utils"] = utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nuscenes2bag"))

from sensor_utils import get_lidar


class TestGetLidar(unittest.TestCase):
    def test_repackages_nuscenes_points_for_autoware(self):
        raw_points = np.array([
            [1.25, -2.5, 3.75, 12.6, 7.6],
            [-4.0, 5.5, -6.25, 300.0, 70000.0],
        ], dtype="<f4")
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_path = Path(temporary_directory)
            lidar_path = data_path / "points.bin"
            raw_points.tofile(lidar_path)
            msg = get_lidar(data_path, {
                "filename": lidar_path.name,
                "timestamp": 1_234_567,
            }, "LIDAR_TOP")

        self.assertEqual(
            [(field.name, field.datatype, field.count, field.offset) for field in msg.fields],
            [
                ("x", PointField.FLOAT32, 1, 0),
                ("y", PointField.FLOAT32, 1, 4),
                ("z", PointField.FLOAT32, 1, 8),
                ("intensity", PointField.UINT8, 1, 12),
                ("return_type", PointField.UINT8, 1, 13),
                ("channel", PointField.UINT16, 1, 14),
            ],
        )
        self.assertEqual((msg.point_step, msg.row_step), (16, 32))
        self.assertEqual((msg.width, msg.height), (2, 1))
        self.assertFalse(msg.is_bigendian)
        self.assertEqual(msg.header.frame_id, "LIDAR_TOP")
        self.assertEqual((msg.header.stamp.sec, msg.header.stamp.nanosec), (1, 234_567_000))

        dtype = np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("intensity", "u1"), ("return_type", "u1"), ("channel", "<u2"),
        ])
        points = np.frombuffer(bytes(msg.data), dtype=dtype)
        np.testing.assert_array_equal(points["x"], raw_points[:, 0])
        np.testing.assert_array_equal(points["y"], raw_points[:, 1])
        np.testing.assert_array_equal(points["z"], raw_points[:, 2])
        np.testing.assert_array_equal(points["intensity"], [13, 255])
        np.testing.assert_array_equal(points["return_type"], [0, 0])
        np.testing.assert_array_equal(points["channel"], [8, 65535])

    def test_rejects_incomplete_raw_record(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_path = Path(temporary_directory)
            lidar_path = data_path / "incomplete.bin"
            np.array([1.0, 2.0, 3.0, 4.0], dtype="<f4").tofile(lidar_path)
            with self.assertRaises(ValueError):
                get_lidar(data_path, {"filename": lidar_path.name, "timestamp": 0}, "LIDAR_TOP")


if __name__ == "__main__":
    unittest.main()
