import copy
import os
import re
import time

from PIL import Image
import pandas as pd
import numpy as np
from torch import nn, optim
from torch.optim import lr_scheduler

from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Dataset
import torch

from matplotlib import pyplot as plt

random_state = 42


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class ImageData(Dataset):

    def __init__(self, data_path, img_folder, img_ids, labels, transform=None):
        self.img_path = os.path.join(data_path, img_folder)
        self.transform = transform
        self.img_ids = img_ids
        self.img_filenames = [f'{im}.jpg' for im in img_ids]
        self.labels = labels

    # Override to give PyTorch size of dataset
    def __len__(self):
            return len(self.img_filenames)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filenames[index]))
        img = img.convert('RGB')
        label = [self.labels[index]]
        return self.transform(img), torch.LongTensor(label)


def get_train_val_test_samples(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, stratify=y, shuffle=True,
                                                        random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train,
                                                      random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)


def load_model(model, device, model_file='best_model.pt'):
    """
    :param model: contains an initiated model (architecture)
    :param model_file: file containing pretrained weights
    :return: model with restored trained weights
    """
    model.load_state_dict(torch.load(model_file))
    model.to(device)
    return model


def train_model(model, dataloaders, criterion, optimizer, num_epochs=15):
    since = time.time()

    history = {'train_loss': [],
               'train_acc': [],
               'val_loss': [],
               'val_acc': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                input_size = inputs.size(0)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(inputs)

                    loss = criterion(outputs, labels.view(input_size))

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * input_size
                running_corrects += torch.sum(preds == labels.squeeze().data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(float(epoch_acc.data.cpu().numpy()))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'best_model.pt')
    return model, history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    return model_ft, input_size


def visualize_model(model, target_column_name, dataloaders, class_names, num_images=8):
    was_training = model.training
    model.eval()
    images_so_far = 0
    n_rows = 2
    n_columns = num_images // n_rows
    fig = plt.figure(figsize=(4 * n_columns, 4.5 * n_rows))
    ax = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax.append(fig.add_subplot(n_rows, n_columns, images_so_far))
                ax[-1].set_title(f'true: {class_names[labels[j]]}\npredicted: {class_names[preds[j]]}')

                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    plt.savefig(f'model_{target_column_name}_predict_result.png')
                    plt.show()
                    return
        model.train(mode=was_training)


def visualize_history(history, target_column_name):
    plt.plot(history['train_acc'], label='Train acc')
    plt.plot(history['train_loss'], label='Train loss')
    plt.plot(history['val_acc'], label='Validation acc')
    plt.plot(history['val_loss'], label='Validation loss')
    plt.title(f'Model for {target_column_name} training history')
    plt.legend()
    plt.savefig(f'{target_column_name} training history.png')
    plt.show()


def get_model_for_target_column(df: pd.DataFrame, ids_column_name: str, target_column_name: str,
                                model_name, num_epochs, batch_size,feature_extract, device):
    target_series = df[target_column_name]
    ids = df[ids_column_name].values
    target = target_series.astype('category').cat.codes.values
    class_names = list(target_series.unique())
    n_classes = len(class_names)

    data_split = get_train_val_test_samples(ids, target)
    X = {'train': data_split[0], 'val': data_split[1], 'test': data_split[2]}
    y = {'train': data_split[3], 'val': data_split[4], 'test': data_split[5]}

    image_datasets = {ds_type: ImageData('.', 'img_n', X[ds_type], y[ds_type], data_transforms[ds_type])
                      for ds_type in ['train', 'val']}

    # Create training and validation dataloaders
    dataloaders_dict = {
        ds_type: torch.utils.data.DataLoader(image_datasets[ds_type], batch_size=batch_size, shuffle=True,
                                             num_workers=4)
        for ds_type in ['train', 'val']
    }

    # imshow(dataloaders_dict['train'][0][0])

    model_ft, input_size = initialize_model(model_name, n_classes, feature_extract, use_pretrained=True)
    # Send the model to GPU
    model_ft = model_ft.to(device)
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)
    print(model_ft)
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)
    print("Model was trained and saved.")
    visualize_history(hist, target_column_name)
    visualize_model(model_ft, target_column_name, dataloaders_dict, class_names)
    # return model_ft, hist

def load_and_test_trained_model(model_file, df: pd.DataFrame, ids_column_name: str, target_column_name: str,
                                model_name, num_epochs, batch_size,feature_extract, device):
    target_series = df[target_column_name]
    ids = df[ids_column_name].values
    target = target_series.astype('category').cat.codes.values
    class_names = list(target_series.unique())
    n_classes = len(class_names)

    data_split = get_train_val_test_samples(ids, target)
    X = {'train': data_split[0], 'val': data_split[1], 'test': data_split[2]}
    y = {'train': data_split[3], 'val': data_split[4], 'test': data_split[5]}

    image_datasets = {ds_type: ImageData('.', 'img_n', X[ds_type], y[ds_type], data_transforms[ds_type])
                      for ds_type in ['train', 'val']}

    # Create training and validation dataloaders
    dataloaders_dict = {
        ds_type: torch.utils.data.DataLoader(image_datasets[ds_type], batch_size=batch_size, shuffle=True,
                                             num_workers=4)
        for ds_type in ['train', 'val']
    }

    # imshow(dataloaders_dict['train'][0][0])

    model_ft, input_size = initialize_model(model_name, n_classes, feature_extract, use_pretrained=False)
    model_ft = load_model(model_ft, device, model_file)
    visualize_model(model_ft, target_column_name, dataloaders_dict, class_names)


if __name__ == '__main__':
    model_name = 'resnet'
    num_epochs = 3
    batch_size = 8
    feature_extract = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    LOAD_TRAINED_MODEL = True

    df = pd.read_excel('products.xlsx')
    df.dropna(inplace=True, subset=['category', 'condition'])

    if LOAD_TRAINED_MODEL:
        model_file = 'best_model_non_feature_extract.pt'
        load_and_test_trained_model(model_file, df, 'id', 'category', model_name, num_epochs, batch_size, feature_extract, device)
    else:
        get_model_for_target_column(df, 'id', 'category', model_name, num_epochs, batch_size, feature_extract, device)


