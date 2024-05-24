import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    inner = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim),
    )
    block = nn.Sequential(nn.Residual(inner), nn.ReLU())
    return block
    raise NotImplementedError()
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    # res_blocks = []
    # block_hidden_dim = hidden_dim
    # for i in range(num_blocks):
    #     res_blocks.append(ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob))
    #     # block_hidden_dim = block_hidden_dim // 2
    
    # model = nn.Sequential(
    #     nn.Linear(dim, hidden_dim),
    #     nn.ReLU(),
    #     *res_blocks,
    #     nn.Linear(hidden_dim, num_classes),
    # )
    blocks = [nn.Linear(dim, hidden_dim),nn.ReLU()]
    for i in range(num_blocks):
        blocks.append(ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob))
    blocks.append(nn.Linear(hidden_dim, num_classes))
    model = nn.Sequential(*blocks)

    return model
    raise NotImplementedError()
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    criteration = nn.SoftmaxLoss()
    if opt:
        model.train()
    else:
        model.eval()
    loss_list = []
    correct_list = []
    for batch in dataloader:
        if opt:
            opt.reset_grad()
        x, y = batch
        y_pred = model(x)
        loss = criteration(y_pred, y)
        if opt:
            loss.backward()
            opt.step()
        loss_list.append(loss.data.numpy())
        pred = np.argmax(y_pred.data.numpy(), axis=1)
        is_correct = pred == y.data
        correct_list.append(is_correct)
    accuracy = np.mean(np.concatenate(correct_list))
    loss_avg = np.mean(loss_list)
    return accuracy, loss_avg
    raise NotImplementedError()
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    test_loader = ndl.data.DataLoader(
        ndl.data.MNISTDataset(f"{data_dir}/t10k-images-idx3-ubyte.gz", f"{data_dir}/t10k-labels-idx1-ubyte.gz"), batch_size=batch_size
    )
    train_loader = ndl.data.DataLoader(
        ndl.data.MNISTDataset(f"{data_dir}/train-images-idx3-ubyte.gz", f"{data_dir}/train-labels-idx1-ubyte.gz"), batch_size=batch_size, shuffle=True
    )
    model = MLPResNet(784, hidden_dim=hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        train_acc, train_loss = epoch(train_loader, model, opt)
        test_acc, test_loss = epoch(test_loader, model)
        print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}, Test Loss {test_loss:.4f}, Test Acc {test_acc:.4f}")
        if epoch == epochs - 1:
            return train_acc, train_loss, test_acc, test_loss
    
    raise NotImplementedError()
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
