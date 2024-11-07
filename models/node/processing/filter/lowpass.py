import abc
from typing import Final, Tuple
from scipy.signal import butter

from models.exception.invalid_parameter_value import InvalidParameterValue
from models.exception.missing_parameter import MissingParameterError
from models.node.processing.filter.filter import Filter


class LowPass(Filter):
    """ This is a lowpass filter. It's a node that filters the input data with a lowpass filter. A lowpass filter
    is a filter that passes frequencies below a certain cutoff value and rejects (attenuates) frequencies above that range.\n
    This class does that by creating a Butterworth scipy filter with the given parameters and btype='Low'. The filtering 
    itself is done by the parent class Filter in the _process method.

    Attributes:
        _MODULE_NAME (`str`): The name of the module (in his case ``node.processing.filter.lowpass``)
    
    configuration.json usage:
        **module** (*str*): The name of the module (``node.processing.filter``)\n
        **type** (*str*): The type of the node (``LowPass``)\n
        **high_cut_frequency_hz** (*float*): The high cut frequency in Hz.\n

        **buffer_options** (*dict*): Buffer options.\n
            **clear_output_buffer_on_data_input** (*bool*): Whether to clear the output buffer when new data is inserted in the input buffer.\n
            **clear_input_buffer_after_process** (*bool*): Whether to clear the input buffer after processing.\n
            **clear_output_buffer_after_process** (*bool*): Whether to clear the output buffer after processing.\n
    
    """
    _MODULE_NAME: Final[str] = 'node.processing.filter.lowpass'

    def _validate_parameters(self, parameters: dict):
        """ Validates the parameters passed to this node. In this case it checks if the parameters are present and if they
        are of the correct type.

        :param parameters: The parameters passed to this node.
        :type parameters: dict

        :raises MissingParameterError: the ``high_cut_frequency_hz`` parameter is required.
        :raises MissingParameterError: the ``order`` parameter is required.
        :raises InvalidParameterValue: the ``high_cut_frequency_hz`` parameter must be a number.
        :raises InvalidParameterValue: the ``order`` parameter must be an int.
        """
        if 'high_cut_frequency_hz' not in parameters:
            raise MissingParameterError(module=self._MODULE_NAME,name=self.name,
                                        parameter='high_cut_frequency_hz')
        if 'order' not in parameters:
            raise MissingParameterError(module=self._MODULE_NAME,name=self.name,
                                        parameter='order')

        if type(parameters['high_cut_frequency_hz']) is not float and type(
                parameters['high_cut_frequency_hz']) is not int:
            raise InvalidParameterValue(module=self._MODULE_NAME,name=self.name,
                                        parameter='high_cut_frequency_hz',
                                        cause='must_be_number')

        if type(parameters['order']) is not int:
            raise InvalidParameterValue(module=self._MODULE_NAME,name=self.name,
                                        parameter='order',
                                        cause='must_be_int')

    @abc.abstractmethod
    def _initialize_parameter_fields(self, parameters: dict):
        """ Initializes the parameter fields of this node.
        """
        super()._initialize_parameter_fields(parameters)

    def _get_filter_coefficients(self, parameters: dict, sampling_frequency_hz: float) -> Tuple[list, list]:
        """ Returns the filter coefficients for the given parameters and sampling frequency. In this case it returns the
        coefficients of a Butterworth filter with the given parameters and btype='low'.

        :param parameters: The parameters passed to this node.
        :type parameters: dict
        :param sampling_frequency_hz: The sampling frequency in Hz.
        :type sampling_frequency_hz: float

        :return: a scipy Butterworth filter with the given parameters and btype='low'.
        :rtype: Tuple[list, list]
        """
        return butter(
            parameters['order'],
            parameters['high_cut_frequency_hz'],
            fs=sampling_frequency_hz,
            btype='low',
            output='ba'
        )