class EMERGELoaderConfig:
    """
    Configuration for loading the EMERGE dataset.

    This is intentionally a plain data container.
    It can later be populated by Tap / argparse / Hydra.
    """

    def __init__(
            self,
            *,
            input_dataset_path: str,
            should_add_predictions: bool = True,
    ):
        self.input_dataset_path = input_dataset_path
        self.should_add_predictions = should_add_predictions

    @classmethod
    def from_dict(cls, cfg: dict):
        """
        Backward-compatible constructor from an existing dict config.
        """
        return cls(
            input_dataset_path=cfg['input_dataset_path'],
            should_add_predictions=cfg.get('should_add_predictions', True)
        )
