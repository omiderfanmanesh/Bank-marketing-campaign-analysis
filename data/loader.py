from data.dataset.bank import Bank


def load(cfg):

    address = cfg.DATASET.DATASET_ADDRESS
    target = cfg.DATASET.TARGET
    dataset_description_file = cfg.DATASET.DATASET_BRIEF_DESCRIPTION

    bank = Bank(dataset_address=address, target=target, dataset_description_file=dataset_description_file)
    return bank
