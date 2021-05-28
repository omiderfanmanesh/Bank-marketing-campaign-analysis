from configs import cfg

if __name__ == '__main__':
    for c in cfg.ENCODER:
        print(c, cfg.ENCODER[c])
