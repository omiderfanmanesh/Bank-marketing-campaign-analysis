import BasedDataset


class Bank(BasedDataset):
    def __init__(self, address, target):
        super(Bank, self).__init__(True, address, target)
