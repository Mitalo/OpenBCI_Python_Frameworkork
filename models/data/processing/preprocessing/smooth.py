from brainflow import DataFilter, AggOperations

from models.data.processing.processing_node import ProcessingNode


class Smooth(ProcessingNode):

    def __init__(self, type: str, period: int) -> None:
        if type is None:
            raise Exception('preprocessing.smooth.invalid.parameters.must.have.type')
        try:
            self._type = type
            self._operation: AggOperations = AggOperations[self._type]
        except KeyError:
            raise KeyError('preprocessing.smooth.invalid.parameters.type.invalid.%s' % self._type)
        if period is None:
            raise Exception('preprocessing.smooth.invalid.parameters.must.have.period')
        if period <= 0:
            raise Exception('preprocessing.smooth.invalid.parameters.period.must.be.greater.than.zero')
        super().__init__({'type': type})
        self._period: int = period

    @classmethod
    def from_config_json(cls, parameters: dict):
        if 'type' not in parameters:
            raise Exception('preprocessing.smooth.invalid.parameters.must.have.type')
        if 'period' not in parameters:
            raise Exception('preprocessing.smooth.invalid.parameters.must.have.period')
        return cls(
            type=parameters['type'],
            period=parameters['period']
        )

    def process(self, data):
        for channel in data:
            DataFilter.perform_rolling_filter(channel, self._period, self._operation)
