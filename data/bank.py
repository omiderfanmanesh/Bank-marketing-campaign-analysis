from .based_dataset import BasedDataset
from utils.FileType import FileType


class Bank(BasedDataset):
    def __init__(self, dataset_address, target, dataset_description_file):
        super(Bank, self).__init__(dataset_type=FileType.CSV, dataset_address=dataset_address, target=target,
                                   dataset_description_file=dataset_description_file)
