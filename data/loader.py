from data.bank import Bank


def load(cfg):
    bank = Bank(cfg=cfg)
    return bank
