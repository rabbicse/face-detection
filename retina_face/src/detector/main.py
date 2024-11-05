# This is a sample Python script.
import torch


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t1 = torch.rand(10).to(device)
    print(t1)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
