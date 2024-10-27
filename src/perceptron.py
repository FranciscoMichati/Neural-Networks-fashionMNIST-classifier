# Standard library imports
import os
from typing import Tuple, Dict

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# PyTorch imports
import torch
from torch import nn
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
import torch.utils.data
from torchvision import datasets
from torchvision.transforms import ToTensor


class MultiLayerPerceptron(nn.Module):
    def __init__(self, n_hidden_1: int, dropout_prob: float) -> None:
        super().__init__()
        self.dropout_prob = dropout_prob
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, n_hidden_1),
            nn.Dropout(self.dropout_prob),
            nn.ReLU(),
            nn.Linear(n_hidden_1, 10),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x


def train_loop(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fun: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    verbose_each: int = 32,
) -> float:
    """
    Training loop for one epoch.

    Args:
        dataloader: DataLoader for training data
        model: PyTorch model to be trained
        loss_fun: Loss function
        optimizer: Optimizer
        device: Device to run on
        verbose_each: Frequency of logging

    Returns:
        avg_train_loss: Average training loss over the epoch
    """
    num_samples = len(dataloader.dataset)  # Number of samples
    num_batches = len(dataloader)  # Number of batches per epoch
    sum_train_loss = 0.0

    model.train()
    model = model.to(device)

    for batch_idx, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fun(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_batch = loss.item()
        sum_train_loss += loss_batch

        # Print progress
        if batch_idx % verbose_each == 0:
            sample = batch_idx * len(X)
            print(
                f"batch={batch_idx} loss={loss_batch:>7f}  processed_samples:[{sample:>5d}/{num_samples:>5d}]"
            )

    avg_train_loss = sum_train_loss / num_batches
    return avg_train_loss


def test_loop(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fun: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Computes the average loss and fraction of correct predictions on the test set.

    Args:
        dataloader: A DataLoader instance containing the test data
        model: A PyTorch model instance
        loss_fun: A PyTorch loss function instance
        device: Device to run on

    Returns:
        A tuple containing the average loss and fraction of correct predictions
    """
    num_samples = len(dataloader.dataset)
    num_batches = len(dataloader)
    total_loss = 0.0
    total_correct = 0

    model.eval()
    model = model.to(device)

    with torch.inference_mode():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            loss = loss_fun(pred, y)

            total_loss += loss.item()
            total_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    avg_loss = total_loss / num_batches
    frac_correct = total_correct / num_samples

    print(
        f"Test Error: Avg loss: {avg_loss:>8f}, Accuracy: {frac_correct * 100:.2f}% \n"
    )
    return avg_loss, frac_correct


def plot_img(data: torch.utils.data.Dataset, idx: int) -> None:
    """
    Plot an example from the dataset.

    Args:
        data: The dataset containing images and labels
        idx: The index of the image to plot
    """
    figure = plt.figure(figsize=(4, 4))
    img, label = data[idx]
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


def show_prediction(
    model: nn.Module,
    example_idx: int,
    test_data: torch.utils.data.Dataset,
    device: torch.device,
    labels_map: Dict[int, str],
) -> None:
    """
    Show prediction for an example from the test data.

    Args:
        model: Trained PyTorch model
        example_idx: Index of the example to show
        test_data: Dataset containing test data
        device: Device to run on
        labels_map: Mapping from label indices to label names
    """
    model.eval()
    x, y = test_data[example_idx][0], test_data[example_idx][1]
    x = x.to(device)
    with torch.no_grad():
        pred = model(x.unsqueeze(0))
    sorted_preds = pred.sort()
    values = softmax(sorted_preds.values[0], dim=-1)
    indices = sorted_preds.indices[0]
    print(f"Correct label: {labels_map[y]}", end="\n----------------\n")
    print("Label         Probability")
    for v, idx in list(zip(values, indices))[::-1]:
        label_pred = labels_map[idx.item()]
        print(f"{label_pred:13}{v.item():.5f}")


def main() -> None:
    # Setup device-agnostic code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Fix torch seed for reproducibility
    torch.manual_seed(40)

    # Download training dataset
    training_data = datasets.FashionMNIST(
        root="data",  # Download Folder
        train=True,  # Choose training set
        download=True,  # Download it
        transform=ToTensor(),  # Transform it to tensor and normalize it
    )

    # Download test dataset
    test_data = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=ToTensor()
    )

    # Labels
    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }

    # Loss function
    loss_fun = nn.CrossEntropyLoss()

    # Merge both datasets
    dataset = torch.utils.data.ConcatDataset([training_data, test_data])

    num_epochs = 50
    batch_size = 500
    k_fold = 5

    splits = KFold(n_splits=k_fold, shuffle=True, random_state=40)

    data = []
    data_cross = []
    n_hidden_vec = [2048, 1024, 512, 256]
    dropout_prob_vec = [0.2, 0.4]
    alpha = 0.01
    model_path = "model_perceptron"

    model_dir = "../out/save_models"
    os.makedirs(model_dir, exist_ok=True)
    dataframes_dir = "../out/save_dataframes"
    os.makedirs(dataframes_dir, exist_ok=True)

    # Main loop
    for n in n_hidden_vec:
        for dropout_prob in dropout_prob_vec:
            fold_val_losses = []

            for fold, (train_idx, val_idx) in enumerate(splits.split(dataset)):
                train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
                test_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

                train_loader = DataLoader(
                    dataset, batch_size=batch_size, sampler=train_sampler
                )
                test_loader = DataLoader(
                    dataset, batch_size=batch_size, sampler=test_sampler
                )

                model = MultiLayerPerceptron(n, dropout_prob)
                model.to(device)

                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=0.001,
                    eps=1e-08,
                    weight_decay=alpha,
                    amsgrad=False,
                )

                # Loop over epochs
                for epoch in range(num_epochs):
                    # Print epoch
                    if epoch % 5 == 0:
                        print(
                            f"n_hidden: {n}, dropout_prob: {dropout_prob}, Fold: {fold+1}, Epoch: {epoch}"
                        )

                    # Training loop
                    train_loss = train_loop(
                        train_loader, model, loss_fun, optimizer, device
                    )
                    train_loss, train_fraction_correct = test_loop(
                        train_loader, model, loss_fun, device
                    )
                    validation_loss, validation_fraction_correct = test_loop(
                        test_loader, model, loss_fun, device
                    )
                    # Append info to data array for the dataframe
                    data.append(
                        np.round(
                            [
                                n,
                                dropout_prob,
                                fold + 1,
                                epoch,
                                train_loss,
                                train_fraction_correct,
                                validation_loss,
                                validation_fraction_correct,
                            ],
                            6,
                        )
                    )

                # Save model parameters
                torch.save(
                    model.state_dict(),
                    f"{model_dir}/{model_path}_{fold}_{n}_{dropout_prob}.pth",
                )

                fold_val_losses.append(validation_loss)

            fold_val_loss = sum(fold_val_losses) / k_fold
            data_cross.append([n, dropout_prob, fold_val_loss])

    columns_names = [
        "n",
        "dropout_prob",
        "k",
        "epoch",
        "train loss",
        "train accuracy",
        "validation loss",
        "validation accuracy",
    ]

    perceptron_df_path = "perceptron_df"
    df = pd.DataFrame(data, columns=columns_names)
    df.to_pickle(f"{dataframes_dir}/{perceptron_df_path}.pkl")

    perceptron_cv_errors_path = "perceptron_cv_dataframe"
    df_cross_errors = pd.DataFrame(
        data_cross, columns=["n", "dropout_prob", "Final model validation loss"]
    )
    df_cross_errors.to_pickle(f"{dataframes_dir}/{perceptron_cv_errors_path}.pkl")


    show_prediction(model, example_idx=0, test_data=test_data, device=device, labels_map=labels_map)
    plot_img(test_data, idx=0)


if __name__ == "__main__":
    main()
