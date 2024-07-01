import sys

sys.dont_write_bytecode = True


from easydict import EasyDict
import logging

logger = logging.getLogger(__name__)


class DataLoaderWrapper:
    def __init__(self, config):
        self.config = config

        self.data_loaders = EasyDict(
            {
                "train": {},
                "valid": {},
                "test": {},
            }
        )

    def set_io(self, io):
        self.io = io

    def build_dataset(self):
        self.data = EasyDict()

        dataset_module_list = self.config.data_loader.dataset_modules
        for module_config in dataset_module_list:
            module_type = module_config.type
            logger.info("Loading dataset module: {}".format(module_config))
            loading_func = getattr(self, module_type)
            loading_func(module_config)
            print("data columns: {}".format(self.data.keys()))
