import unittest
from models.node.processing.filter.highpass import HighPass
from models.framework_data import FrameworkData

class TestHighPassFilter(unittest.TestCase):
    def test_highpass_filter(self):
        parameters = {
            'order': 5,
            'low_cut_frequency_hz': 20.0
        }
        sampling_frequency_hz = 250.0
        data = {
            'channel1': FrameworkData(sampling_frequency_hz=sampling_frequency_hz)
        }
        data['channel1'].input_data_on_channel([1, 2, 3, 4, 5], 'channel1')

        highpass_filter = HighPass(parameters)
        filtered_data = highpass_filter._process(data)

        # Verifique se os dados filtrados estão corretos
        self.assertIsNotNone(filtered_data)
        self.assertIn('channel1', filtered_data)
        self.assertTrue(len(filtered_data['channel1'].get_data_on_channel('channel1')) > 0)

if __name__ == '__main__':
    unittest.main()