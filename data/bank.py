#  Copyright (c) 2021.
#

from data.based.based_dataset import BasedDataset
from data.based.file_types import FileTypes


class Bank(BasedDataset):
    def __init__(self, cfg):
        super(Bank, self).__init__(cfg=cfg, dataset_type=FileTypes.CSV)
