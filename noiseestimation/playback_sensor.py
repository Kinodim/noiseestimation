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
        control_fields (list, optional): List of fields the sensor should output
            without added noise
    """
    def __init__(self, data_filename, fields, control_fields=None):
        self.fields = fields
        if not hasattr(fields, '__iter__'):
            self.fields = [fields]
        self.control_fields = control_fields
        if control_fields is not None and \
                not hasattr(control_fields, '__iter__'):
            self.control_fields = [control_fields]
        self.index = 0
        self.__read_data(data_filename)
        rnd.seed()

    def read(self, R=[[0]]):
        """Outputs the current data entry

        Args:
            R (ndarray, optional): Desired measurement covariance matrix.
                Defaults to zero.

        Returns:
            tuple:
                - int: Timestamp of the reading
                - ndarray: stacked array of measurement and control readings

        Raises:
            IndexError: No more data is available
        """
        if self.index >= len(self.data):
            raise IndexError("No more data available")

        # time
        try:
            time = float(self.data[self.index]["time"])
        except KeyError:
            print("No time field found")
            time = 0
        except ValueError:
            print("Error parsing current time")
            time = 0

        # measurements
        measurements = np.zeros((len(self.fields), 1), "double")
        for field_idx, field in enumerate(self.fields):
            parsed_val = self.__parse_value(field)
            measurements[field_idx, 0] = parsed_val

        # controls
        if self.control_fields is None:
            controls = np.zeros((0, 1))
        else:
            controls = np.zeros((len(self.control_fields), 1), "double")
            for field_idx, field in enumerate(self.control_fields):
                parsed_val = self.__parse_value(field)
                controls[field_idx, 0] = parsed_val

        noise = rnd.multivariate_normal(np.zeros(len(R)), R).reshape(-1, 1)
        measurements = measurements + noise
        y = np.vstack((measurements, controls))

        self.index += 1
        return time, y

    def __read_data(self, filename):
        path = os.path.join(os.getcwd(), filename)
        if not os.path.isfile(path):
            print("Could not find file %s" % path)
            return
        with open(path, "r") as f:
            content = f.read()
        self.data = json.loads(content)

    def __parse_value(self, field):
        parsed_val = 0
        try:
            parsed_val = float(self.data[self.index][field])
        except ValueError:
            print("Error parsing string: %s" % self.data[self.index][field])
            print("Returning default value of zero")

        return parsed_val
