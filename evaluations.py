import torch
import torch.nn as nn
from utils.optim import csi5140Adam, csi5140GDM, csi5140GD
from utils.regularization import csi5140_cosine_learning_rate_decay as csi5140_cosine
def check_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _, predictions = scores.max(1)

            num_correct += (predictions == y).sum().item()
            num_samples += predictions.size(0)
    
    acc = num_correct / num_samples * 100

    model.train()
    return acc  

def train_model(
    model,
    train_loader,
    test_loader,
    device,
    epochs=5,
    lr=0.01,
    optimizer_type="adam",
    weight_decay=0.0,       # L2
    betas=(0.9, 0.999),     # Adam
    momentum=0.9,           # SGD
    learn_rate_type=None,    
    step_size=2,             # Step learning rate decay
    lr_min = 0.001,          # min learning rate during decay
    lr_max = 0.01,           # max learning rate during decay
    gamma=0.9                # Exponential 
):
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_type = (optimizer_type or "").lower()
    learn_rate_type = (learn_rate_type or "").lower()

    # ---- Optimizers ----
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay
        )
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    elif optimizer_type == "csi5140_adam":
            optimizer = csi5140Adam(
            model.parameters(),
            lr=lr,
            betas=betas,
            #weight_decay=weight_decay
        )
    elif optimizer_type == "csi5140_gdm":
            optimizer = csi5140GDM(
            model.parameters(),
            momentum=momentum,
            lr=lr,
            #weight_decay=weight_decay
        )


    # ---- Learning Rate Decay ----
    if learn_rate_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif learn_rate_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
    elif learn_rate_type == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=gamma
    )
    elif learn_rate_type == "csi5140_cosine":
        scheduler = None

    else:
        scheduler = None

    train_accs = []
    test_accs = []
    train_costs = []
    iteration = 0
    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        epoch_loss = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad() # Set gradients back to 0

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward() # Backprop
            optimizer.step() # Update weights

            #csi5140 learning rate decays
            if learn_rate_type == "csi5140_cosine":
                #this function will modify learning rates
                csi5140_cosine(optimizer, epoch, epochs, lr_max, lr_min)

            train_costs.append((iteration,round(loss.item(), 4)))
            epoch_loss += loss.item()
            iteration += 1
            
            _, predicted = torch.max(outputs, 1) # Get best class for every sample
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        test_acc = check_accuracy(test_loader, model, device)
        avg_loss = epoch_loss / len(train_loader)

        train_accs.append(round(train_acc, 4))
        test_accs.append(round(test_acc, 4))

        if scheduler:
            scheduler.step()

        print(f"Epoch {epoch+1}: Loss {avg_loss:.4f}, Train {train_acc:.2f}%, Test {test_acc:.2f}%")

    return model, train_accs, test_accs, train_costs

import matplotlib.pyplot as plt

def plot_metrics(train_accs, test_accs, costs, title_prefix="Model"):
    # --- Accuracy Plot ---
    plt.figure()
    plt.plot(range(1, len(train_accs) + 1), train_accs, label="Train Accuracy")
    plt.plot(range(1, len(test_accs) + 1), test_accs, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title_prefix} Accuracy vs Epochs")
    plt.legend()
    plt.grid()
    plt.show()

    # --- Cost Plot ---
    plt.figure()
    plt.plot(range(1, len(costs) + 1), costs)
    plt.xlabel("Iteration")
    plt.ylabel("Cost (Loss)")
    plt.title(f"{title_prefix} Cost vs Iterations")
    plt.grid()
    plt.show()