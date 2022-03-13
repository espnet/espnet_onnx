from abc import ABC


class AbsModel(ABC):
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def get_input_names(self):
        raise NotImplementedError

    def get_output_names(self):
        raise NotImplementedError

    def get_dynamix_axes(self):
        return {}

    def get_model_config(self):
        raise NotImplementedError
