import mne as mne

from models.data.processing.feature_extraction.feature_extractor import FeatureExtractor


class CSP(FeatureExtractor):

    def __init__(self, n_components: int) -> None:
        super().__init__({'type': 'CSP'})
        if n_components <= 0:
            raise ValueError(
                'processing.trainable.feature.extractor.csp.parameters.n_components.must.be.greater.than.zero')
        self.n_components = n_components
        self.csp = mne.decoding.csp.CSP(n_components=self.n_components)

    @classmethod
    def from_config_json(cls, parameters: dict):
        super().from_config_json(parameters)
        if 'n-components' not in parameters['parameters']:
            raise ValueError('processing.trainable.feature.extractor.csp.parameters.must.have.n-components')
        return cls(
            n_components=parameters['parameters']['n-components']
        )

    def process(self, data):
        extracted = self.csp.transform(data)
        for component in range(self.n_components):
            data[len(data) + component] = extracted[component]

    def train(self, data, label):
        self.csp = self.csp.fit(data, label)