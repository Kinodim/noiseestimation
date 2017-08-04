import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
import os.path
import noiseestimation.playback_sensor as ps


class TestPlaybackSensor:
    def setup(self):
        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "data",
                                "vehicle_state.json")
        self.sensor = ps.PlaybackSensor(filename,
                                        ["fVx", "fYawrate", "fStwAng"])
        self.num_entries = len(self.sensor.data)

    def test_read_data__existent(self):
        assert hasattr(self.sensor, "data")

    def test_read_data__nonexistent(self):
        sensor = ps.PlaybackSensor("does_not_exist", ["field"])
        assert not hasattr(sensor, "data")

    def test_read__first_entry(self):
        time, fields = self.sensor.read([[0]])
        assert time == pytest.approx(1484917519047638000)
        assert_array_almost_equal(fields,
                                  np.array([[0.],
                                            [0.00820305],
                                            [4.53611]]))

    def test_read__fields(self):
        for _ in range(self.num_entries):
            time, fields = self.sensor.read([[0]])
            assert len(fields) == 3

    def test_read__end_of_data(self):
        for elem in range(self.num_entries):
            _, _ = self.sensor.read([[0]])

        with pytest.raises(IndexError):
            self.sensor.read([[0]])

    def test_read__no_time(self):
        self.sensor.data[0].pop("time")
        time, _ = self.sensor.read([[0]])
        assert time == 0
