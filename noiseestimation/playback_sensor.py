import json
import os
import numpy as np
import numpy.random as rnd


class PlaybackSensor:
    """Facilitates reading stored sensor data

    Args:
        data_filename (string): Filename of the json data file
        fields (list): List of fields the sensor should output. The 'time' field
            will be read by default
    """
    def __init__(self, data_filename, fields):
        self.fields = fields
        if not hasattr(fields, '__iter__'):
            self.fields = [fields]
        self.index = 0
        self.read_data(data_filename)

    def read_data(self, filename):
        path = os.path.join(os.getcwd(), filename)
        if not os.path.isfile(path):
            print("Could not find file %s" % path)
            return
        with open(path, "r") as f:
            content = f.read()
        self.data = json.loads(content)

    def read(self, R):
        if self.index >= len(self.data):
            raise IndexError("No more data available")

        try:
            time = float(self.data[self.index]["time"])
        except KeyError:
            print("No time field found")
        except ValueError:
            print("Error parsing current time")

        y = np.zeros((len(self.fields), 1), "double")
        for field_idx, field in enumerate(self.fields):
            parsed_val = 0
            try:
                parsed_val = float(self.data[self.index][field])
            except ValueError:
                print("Error parsing string: %s" % self.data[self.index][field])
                print("Returning default value of zero")
            y[field_idx, 0] = parsed_val

        noise = rnd.multivariate_normal(np.zeros(len(R)), R).reshape(-1, 1)
        self.index += 1
        return time, y + noise
