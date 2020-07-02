import yaml

class Config(dict):
    """
    Holds the configuration for anything you want it to.
    """

    def __init__(self, config_dict, base=None):
        if base:
            super().__init__(base)
        else:
            super().__init__()
        self.update(config_dict)

    @classmethod
    def load(klass, config_path, base_path=None):
        with open(config_path, 'r') as c:
            config = yaml.safe_load(c)
        base = None
        if base_path:
            with open(base_path, 'r') as b:
                base = yaml.safe_load(b)
        return  klass(config, base=base)
