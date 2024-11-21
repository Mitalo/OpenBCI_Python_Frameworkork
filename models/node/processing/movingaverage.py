import numpy as np
from typing import List, Dict, Final
from models.framework_data import FrameworkData
from models.node.processing.processing_node import ProcessingNode
from models.exception.missing_parameter import MissingParameterError
from models.exception.invalid_parameter_value import InvalidParameterValue

class MovingAverage(ProcessingNode):
    """ This node computes the moving average of the input signal with the specified window size.

    Attributes:
        _MODULE_NAME (str): The name of the module (in this case ``node.processing.movingaverage``)
    """
    _MODULE_NAME: Final[str] = 'node.processing.movingaverage'

    INPUT_MAIN: Final[str] = 'main'
    OUTPUT_MAIN: Final[str] = 'main'

    def __init__(self, parameters: dict):
        super().__init__(parameters)
        self._validate_parameters(parameters)
        self.window_size = parameters['window_size']

    def _is_next_node_call_enabled(self) -> bool:
        """ This method will return ``True`` if the next node call is enabled. This method will always return ``True``
        because the next node call is always enabled.
        """
        return True

    def _is_processing_condition_satisfied(self) -> bool:
        """ This method will return ``True`` if the processing condition is satisfied. This method will return ``True`` if the input buffer has data.

        :return: ``True`` if the input buffer has data, ``False`` otherwise.
        :rtype: bool
        """
        return self._input_buffer[self.INPUT_MAIN].get_data_count() > 0

    def _validate_parameters(self, parameters: dict):
        """ Validates the parameters passed to this node. In this case it checks if the parameters are present and if they
        are of the correct type.

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
        
    def _process(self, data: Dict[str, FrameworkData]) -> Dict[str, FrameworkData]:
        """ This method processes the data that was inputted to the node. It computes the moving average of the input signal.

        :param data: The data that was inputted to the node.
        :type data: Dict[str, FrameworkData]

        :return: The smoothed data.
        :rtype: Dict[str, FrameworkData]
        """
        input_data = data[self.INPUT_MAIN]
        smoothed_data = FrameworkData(sampling_frequency_hz=input_data.sampling_frequency,
                                      channels=input_data.channels)

        for channel in input_data.channels:
            signal = input_data.get_data_on_channel(channel)
            smoothed_signal = np.convolve(signal, np.ones(self.window_size) / self.window_size, mode='same')
            smoothed_data.input_data_on_channel(smoothed_signal, channel)

        return {
            self.OUTPUT_MAIN: smoothed_data
        }
    
    def _get_inputs(self) -> List[str]:
        """ This method will return the inputs of the node.
        
        :return: The inputs of the node.
        :rtype: list
        """
        return [
            self.INPUT_MAIN
        ]

    def _get_outputs(self) -> List[str]:
        """ This method will return the outputs of the node.
        
        :return: The outputs of the node.
        :rtype: list
        """
        return [
            self.OUTPUT_MAIN
        ]