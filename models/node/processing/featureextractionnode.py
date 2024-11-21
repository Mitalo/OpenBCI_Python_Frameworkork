import numpy as np
from typing import Dict, List, Final
from models.framework_data import FrameworkData
from models.node.processing.processing_node import ProcessingNode
from models.exception.missing_parameter import MissingParameterError
from models.exception.invalid_parameter_value import InvalidParameterValue

class FeatureExtractionNode(ProcessingNode):
    """ This node extracts features from the input signal and associates them with event labels.

    Attributes:
        _MODULE_NAME (str): The name of the module (in this case ``node.processing.feature_extraction_node``)
    """
    _MODULE_NAME: Final[str] = 'node.processing.feature_extraction_node'

    INPUT_FEATURES: Final[str] = 'features'
    INPUT_EVENTS: Final[str] = 'events'

    OUTPUT_FEATURES: Final[str] = 'features'
    OUTPUT_EVENTS: Final[str] = 'events'

    def __init__(self, parameters: dict):
        super().__init__(parameters)
        self.fs = parameters.get('sampling_frequency', 250)
        self.window_size = parameters.get('window_size', 50)
        self._validate_parameters(parameters)

    def _validate_parameters(self, parameters: dict):
        """ Validates the parameters passed to this node.

        :param parameters: The parameters to validate.
        :type parameters: dict
        """
                
        if 'window_size' not in parameters:
            raise MissingParameterError(module=self._MODULE_NAME,name=self.name,
                                        parameter='window_size')
        
        if type(parameters['window_size']) is not float and type(
            parameters['window_size']) is not int:
            raise InvalidParameterValue(module=self._MODULE_NAME,name=self.name,
                                        parameter='window_size',
                                        cause='must_be_number')
         
        if parameters['window_size'] <= 1:
            raise InvalidParameterValue(module=self._MODULE_NAME, name=self.name, parameter='window_size', cause='must_be_greater_than_0')

    def _is_next_node_call_enabled(self) -> bool:
        """ This method will return ``True`` if the next node call is enabled. This method will always return ``True``
        because the next node call is always enabled.
        """
        return self._output_buffer[self.OUTPUT_EVENTS].get_data_count() > 0 and self._output_buffer[self.OUTPUT_FEATURES].get_data_count() > 0

    def _is_processing_condition_satisfied(self) -> bool:
        """ This method will return ``True`` if the processing condition is satisfied. This method will return ``True`` if the input buffer has data.

        :return: ``True`` if the input buffer has data, ``False`` otherwise.
        :rtype: bool
        """
        return self._input_buffer[self.INPUT_EVENTS].get_data_count() > 0 \
               and self._input_buffer[self.INPUT_EVENTS].get_data_count() >= self._input_buffer[
                   self.INPUT_FEATURES].get_data_count() \
               and self._input_buffer[self.INPUT_FEATURES].get_data_count() > 0

    def _process(self, data: Dict[str, FrameworkData]) -> Dict[str, FrameworkData]:
        """ This method processes the data that was inputted to the node. It extracts features and associates them with event labels.

        :param data: The data that was inputted to the node.
        :type data: Dict[str, FrameworkData]

        :return: The extracted features and associated labels.
        :rtype: Dict[str, FrameworkData]
        """
        signal_data = data['features']
        event_data = data['events']

        features = []
        labels = []

        signal_fp1_data = signal_data.get_data_on_channel('Fp1')
        signal_c3_data = signal_data.get_data_on_channel('C3')
        event_marker_data = event_data.get_data_on_channel('marker')

        # Extrair características para cada janela de tempo
        for i in range(0, len(signal_fp1_data), self.fs):
            window_fp1 = signal_fp1_data[i:i+self.fs]
            window_c3 = signal_c3_data[i:i+self.fs]

            if len(window_fp1) == self.fs and len(window_c3) == self.fs:
                combined_features = self.extract_features(np.concatenate([window_fp1, window_c3]), self.fs)
                features.append(combined_features)
                label = event_data.get_data_on_channel('marker')[i]
                labels.append(label)

        features_data = FrameworkData(sampling_frequency_hz=self.fs, channels=['features'])
        labels_data = FrameworkData(sampling_frequency_hz=self.fs, channels=['labels'])

        features_data.input_data_on_channel(np.array(features), 'features')
        labels_data.input_data_on_channel(np.array(labels), 'labels')

        return {
            self.OUTPUT_FEATURES: features_data,
            self.OUTPUT_EVENTS: labels_data
        }

    def extract_features(self, signal, fs):
        mav = np.mean(np.abs(signal))
        zc = len(np.where(np.diff(np.sign(signal)))[0])
        ssc = sum(np.sign(signal[i] - signal[i-1]) != np.sign(signal[i+1] - signal[i]) for i in range(1, len(signal)-1))
        wl = np.sum(np.abs(np.diff(signal)))
        energy = np.sum(signal**2) / len(signal)
        return [mav, zc, ssc, wl, energy]

    def _get_inputs(self) -> List[str]:
        """ Returns the input fields of this node.
        """
        return [
            self.INPUT_FEATURES,
            self.INPUT_EVENTS
        ]

    def _get_outputs(self) -> List[str]:
        """ Returns the output fields of this node.
        """
        return [
            self.OUTPUT_FEATURES,
            self.OUTPUT_EVENTS
        ]