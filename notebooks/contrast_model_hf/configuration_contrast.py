from transformers import PretrainedConfig

class ContrastConfig(PretrainedConfig):
    model_type = "contrast"

    def __init__(self, image_size: int = 768, **kwargs):
        self.image_size = image_size
        super().__init__(**kwargs)
