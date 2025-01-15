from torch import Tensor
from torch.nn import Module, Conv2d, MaxPool2d, \
    ReLU, Linear, Sequential, Dropout, Softmax

class MyModel(Module):

    def __init__(self):
        """
        Initializes the layers of the Mymodel
        """

        super().__init__()

        self.block1 = Sequential(
            Conv2d(1, 64, kernel_size=(3, 3), stride=(1,1)),
            ReLU(),
            MaxPool2d(kernel_size=(2, 4)),
            Dropout(0.2)
        )

        self.block2 = Sequential(
            Conv2d(64, 64, kernel_size=(3, 5), stride=(1, 1)),
            ReLU(),
            MaxPool2d(kernel_size=(2, 4)),
            Dropout(0.2)
        )

        self.fully_connected = Sequential(
            Linear(13440, 32),
            Dropout(0.2),
            ReLU()
        )

        self.output = Sequential(
            Linear(32, 10)
        )

    def forward(self, x: Tensor) -> Tensor:

        x = x.unsqueeze(1)

        x = self.block1(x)
        x = self.block2(x)

        x = x.view(x.size(0), -1)

        x = self.fully_connected(x)

        x = self.output(x)

        return x

def main():
    model = MyModel()
    print(model)



if __name__ == '__main__':
    main()