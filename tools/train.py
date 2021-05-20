from data import load
from configs import cfg



def main():
    bank = load(cfg)
    bank.load_dataset()
    bank.description()



if __name__ == '__main__':
    main()
