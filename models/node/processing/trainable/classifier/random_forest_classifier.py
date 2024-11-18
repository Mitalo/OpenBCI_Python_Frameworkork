import abc
from typing import Final, Any, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.base import TransformerMixin, BaseEstimator

from models.exception.invalid_parameter_value import InvalidParameterValue
from models.exception.missing_parameter import MissingParameterError
from models.framework_data import FrameworkData
from models.node.processing.trainable.classifier.sklearn_classifier import SKLearnClassifier

class RandomForestSKLearnClassifier(SKLearnClassifier):
    def __init__(self, parameters: dict):
        super().__init__(parameters)
        self._initialize_parameter_fields(parameters)
        self._validate_parameters(parameters)
        self.sklearn_processor = self._initialize_trainable_processor()

    def _initialize_parameter_fields(self, parameters: dict):
        super()._initialize_parameter_fields(parameters)
        self.n_estimators = parameters.get('n_estimators', 100)
        self.random_state = parameters.get('random_state', 42)

    def _validate_parameters(self, parameters: dict):
        super()._validate_parameters(parameters)
        if not isinstance(self.n_estimators, int) or self.n_estimators <= 0:
            raise ValueError("n_estimators must be a positive integer")
        if not isinstance(self.random_state, int):
            raise ValueError("random_state must be an integer")

    def _initialize_trainable_processor(self) -> Tuple[TransformerMixin, BaseEstimator]:
        return RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)

    def _should_retrain(self) -> bool:
        return False

    def _is_next_node_call_enabled(self) -> bool:
        return self._is_trained and self._output_buffer[self.OUTPUT_MAIN].has_data()

    def _format_processed_data(self, processed_data: Any, sampling_frequency: float) -> FrameworkData:
        formatted_data = FrameworkData(sampling_frequency_hz=sampling_frequency)
        formatted_data.input_data_on_channel(processed_data)
        return formatted_data