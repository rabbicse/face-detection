# This is a sample Python script.
from network import Network
from train import Train


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def train():
    print(f'Loading data...')
    train = Train()
    train.load_data()
    # train.show_images()
    train.train_data()
    # train.test_data()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
