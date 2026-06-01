import os
import torch
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def compute_accuracy_multiclass(outputs, targets):
    preds = outputs.argmax(dim=1)
    correct = (preds == targets).sum().item()

    return correct / targets.size(0)


def compute_accuracy_binary_single_neuron(outputs, targets):
    # single neuron output requires sigmoid
    probs = torch.sigmoid(outputs)
    preds = (probs > 0.5).float().squeeze()

    correct = (preds == targets).sum().item()

    return correct / len(targets)



def train(model, dataloader, device, optimizer, epoch, num_epochs, total_steps, loss_fn, compute_accuracy):
    running_loss = 0.0
    running_acc = 0.0
    model.train()

    for i, (data, targets) in enumerate(dataloader):
        data = data.to(device)
        targets = targets.to(device)

        outputs = model(data)

        loss = loss_fn(outputs, targets)
        accuracy = compute_accuracy(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy

        if (i+1) % 50 == 0:
            print(
                f'TRAINING --> Epoch: {epoch+1}/{num_epochs}, ' +
                f'Step: {i+1}/{total_steps}, ' +
                f'Loss: {running_loss / (i+1)}, '
                f'Accuracy: {running_acc / (i+1)}'
            )
    running_loss = running_loss / total_steps
    running_acc = running_acc / total_steps

    return running_loss, running_acc


def validate(model, dataloader, device, epoch, num_epochs, total_steps, loss_fn, compute_accuracy):
        running_loss = 0.0
        running_acc = 0.0
        model.eval()

        with torch.no_grad():
            for i, (data, targets) in enumerate(dataloader):
                data = data.to(device)
                targets = targets.to(device)

                outputs = model(data)

                loss = loss_fn(outputs, targets)
                accuracy = compute_accuracy(outputs, targets)

                running_loss += loss.item()
                running_acc += accuracy

                if (i+1) % 50 == 0:
                    print(
                        f'VALIDATION --> Epoch: {epoch+1}/{num_epochs}, ' +
                        f'Step: {i+1}/{total_steps}, ' +
                        f'Loss: {running_loss / (i+1)}, '
                        f'Accuracy: {running_acc / (i+1)}'
                    )
            running_loss = running_loss / total_steps
            running_acc = running_acc / total_steps

        return running_loss, running_acc


def save_best_model(model, path, val_loss, val_losses, epoch):
    print(path)
    path = path / f'model_state_dict.pt'

    if len(val_losses) == 0:
        torch.save(
            model.state_dict(),
            path
        )
        print(
            'SAVING --> First epoch: \n' +
            f'Val Loss: {val_loss}\n' +
            f'Saving new model to:\n{path}'
        )
    elif val_loss < min(val_losses):
        torch.save(
            model.state_dict(),
            path
        )
        print(
            'SAVING --> Found model with better validation loss: \n' +
            f'New Best Val Loss: {val_loss}\n' +
            f'Old Best Val Loss: {min(val_losses)}\n'
            f'Saving new model to:\n{path}'
        )
    return path


def plot_epoch_metrics(x, y, data_names, title_prefix, yaxis_label, file, disp=False):
    plt.clf()
    for i in y:
        plt.plot(x, i)
    plt.title(title_prefix + ' ' + ' vs. '.join(data_names) + ' ' + yaxis_label)
    plt.ylabel(yaxis_label)
    plt.legend(data_names)
    plt.savefig(file)

    if disp:
        plt.show()


def train_loop(model, arch, train_dataloader, val_dataloader, device, optimizer, num_epochs, loss_fn, compute_accuracy, save_path=Path('./out')):
    print(f'Models will be saved to: {save_path}')
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)

    train_total_steps = len(train_dataloader)
    val_total_steps = len(val_dataloader)

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, train_dataloader, device, optimizer, epoch, num_epochs, train_total_steps, loss_fn, compute_accuracy)
        print(
            f'TRAINING --> Epoch {epoch+1}/{num_epochs} DONE, ' +
            f'Avg Loss: {train_loss}, Avg Accuracy: {train_accuracy}'
        )

        val_loss, val_accuracy = validate(model, val_dataloader, device, epoch, num_epochs, val_total_steps, loss_fn, compute_accuracy)
        print(
            f'VALIDATION --> Epoch {epoch+1}/{num_epochs} DONE, ' +
            f'Avg Loss: {val_loss}, Avg Accuracy: {val_accuracy}'
        )

        new_saved_model_path = save_best_model(model, save_path, val_loss, val_losses, epoch)

        train_losses.append(train_loss)
        train_accs.append(train_accuracy)
        val_losses.append(val_loss)
        val_accs.append(val_accuracy)
    return (train_losses, train_accs), (val_losses, val_accs), new_saved_model_path


def evaluate(model, dataloader, device, total_steps, loss_fn, compute_accuracy):
    running_loss = 0.0
    running_acc = 0.0
    all_preds = []
    all_targets = []
    model.eval()

    with torch.no_grad():
        for i, (data, targets) in enumerate(dataloader):
            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)

            loss = loss_fn(outputs, targets)
            accuracy = compute_accuracy(outputs, targets)

            # TODO: automatically choose different output processing based on binary or multiclass
            #preds = outputs.argmax(dim=1)
            preds = (torch.sigmoid(outputs) > 0.5).long().view(-1)

            running_loss += loss.item()
            running_acc += accuracy

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.long().view(-1).cpu().numpy().tolist())

            if (i+1) % 256 == 0:
                print(
                    f'TEST --> ' +
                    f'Step: {i+1}/{total_steps}, ' +
                    f'Loss: {running_loss / (i+1)}, '
                    f'Accuracy: {running_acc / (i+1)}'
                )
    running_loss = running_loss / total_steps
    running_acc = running_acc / total_steps

    return running_loss, running_acc, all_preds, all_targets


def score(all_preds, all_targets):
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    y_pred = np.asarray(all_preds).reshape(-1)
    y_true = np.asarray(all_targets).reshape(-1)

    cm = confusion_matrix(y_true, y_pred)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return precision, recall, f1, cm


def print_data_iteration(dataloader):
    for i, (data, targets) in enumerate(dataloader):
        plt.plot(data[0])
        plt.title(f'LSTM Input (Smoking - {int(targets[0].numpy())})')
        plt.show()
        return


def plot_conf_matrix(cm, save_path):
    plt.clf()
    plt.imshow(cm)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.savefig(f"{save_path}/confusion_matrix.png")
