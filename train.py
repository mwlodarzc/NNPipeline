import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


class NeuralNetwork(nn.Module):
    def __init__(self, layers: nn.Sequential):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = layers

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class TrainModel():
    def __init__(self, model: NeuralNetwork, dataset_name: str, batch_size: int, loss_fn: str = "CrossEntropyLoss", optimizer: str = "SGD", lr: int = 1e-3):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.training_data = getattr(datasets, dataset_name)(
            root="data", train=True, download=True, transform=ToTensor())
        self.test_data = getattr(datasets, dataset_name)(
            root="data", train=False, download=True, transform=ToTensor())
        self.train_dataloader = DataLoader(
            self.training_data, batch_size=batch_size)
        self.test_dataloader = DataLoader(
            self.test_data, batch_size=batch_size)

        self.model = model.to(self.device)

        self.loss_fn = getattr(nn, loss_fn)()
        self.optimizer = getattr(torch.optim, optimizer)(
            self.model.parameters(), lr=lr)

    def run(self, epochs: int):
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.train()
            self.test()

    def train(self):
        size = len(self.train_dataloader.dataset)
        self.model.train()
        for batch, (X, y) in enumerate(self.train_dataloader):
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(self):
        size = len(self.test_data)
        num_batches = len(self.test_dataloader)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in self.test_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(
            f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def eval(self, classes: list):
        self.model.eval()
        x, y = self.test_data[0][0], self.test_data[0][1]
        with torch.no_grad():
            pred = self.model(x)
            predicted, actual = classes[pred[0].argmax(0)], classes[y]
            print(f'Predicted: "{predicted}", Actual: "{actual}"')

    def get_data(self):
        return self.test_data, self.test_dataloader

    def get_device(self):
        return self.device

    def get_model(self):
        return self.model

    def save(self, path: str):
        torch.save(self.model.state_dict(), f"{path}.pth")

    def load(self, path: str):
        self.model.load_state_dict(torch.load(f"{path}.pth"))


if __name__ == '__main__':
    layers = nn.Sequential(
        nn.Linear(28*28, 5_000),
        nn.ReLU(),
        nn.Linear(5_000, 3_500),
        nn.ReLU(),
        nn.Linear(3_500, 3_000),
        nn.ReLU(),
        nn.Linear(3_000, 1_000),
        nn.ReLU(),
        nn.Linear(1_000, 10)
    )
    model = NeuralNetwork(layers=layers)
    train = TrainModel(model, "FashionMNIST", 64)
    train.run(10)
    train.eval(classes=[
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ])
