import copy
import os
import re
import time
from typing import List

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


def load_model(model, device, model_file):
    """
    :param model: contains an initiated model (architecture)
    :param model_file: file containing pretrained weights
    :return: model with restored trained weights
    """
    model.load_state_dict(torch.load(model_file))
    model.to(device)
    return model


def train_model(model, dataloaders, criterion, optimizer, save_model_name, num_epochs=15):
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
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                input_size = inputs.size(0)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.view(input_size))
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * input_size
                running_corrects += torch.sum(preds == labels.squeeze().data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(float(epoch_acc.data.cpu().numpy()))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), save_model_name)
    return model, history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # In case we will try other architectures, we pass 'model_name' as eah model
    # requires special treatment during initializations
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    return model_ft, input_size


def visualize_model(model, dataloader, class_names, im_name, num_images=8):
    was_training = model.training
    model.eval()
    images_so_far = 0
    n_rows = 2
    n_columns = num_images // n_rows
    fig = plt.figure(figsize=(4 * n_columns, 4.5 * n_rows))
    ax = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
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
                    plt.savefig(im_name)
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
                                model_name, num_epochs, batch_size, feature_extract, device):
    target_series = df[target_column_name].astype('category')
    ids = df[ids_column_name].values
    target = [values_dict[target_column_name][val] for val in target_series.values]
    class_names = list(values_dict[target_column_name].keys())
    n_classes = len(class_names)

    data_split = get_train_val_test_samples(ids, target)
    X = {'train': data_split[0], 'val': data_split[1], 'test': data_split[2]}
    y = {'train': data_split[3], 'val': data_split[4], 'test': data_split[5]}
    # save test ids and lables
    test_class_names = [ids_dict[target_column_name][i] for i in y['test']]
    pd.DataFrame({'id': X['test'], 'target': y['test'], target_column_name: test_class_names}) \
        .to_csv(f'{target_column_name}_test_sample.csv', index=False)

    image_datasets = {ds_type: ImageData('.', 'img_n', X[ds_type], y[ds_type], data_transforms[ds_type])
                      for ds_type in ['train', 'val']}

    # Create training and validation dataloaders
    dataloaders_dict = {
        ds_type: torch.utils.data.DataLoader(image_datasets[ds_type], batch_size=batch_size, shuffle=True,
                                             num_workers=4)
        for ds_type in ['train', 'val']
    }

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
    feature_extract_str = 'feature_extract' if feature_extract else 'finetuning'
    save_model_name = f'best_model_{target_column_name}_{feature_extract_str}.pt'
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, save_model_name,num_epochs=num_epochs)
    print("Model was trained and saved.")
    visualize_history(hist, target_column_name)
    visualize_model(model_ft, dataloaders_dict['val'], class_names, f'model_{target_column_name}_predict_result.png')


def load_and_test_trained_model(model_file, num_images, ids_column_name: str, target_column_name: str,
                                model_name, batch_size, feature_extract, device):
    test_df = pd.read_csv(f'{target_column_name}_test_sample.csv')
    ids = test_df[ids_column_name].values
    y = test_df['target']  # type: List[int]
    class_names = list(test_df[target_column_name].unique())
    n_classes = len(class_names)

    image_datasets = ImageData('.', 'img_n', ids, y, data_transforms['val'])
    dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle=True, num_workers=4)

    model_ft, input_size = initialize_model(model_name, n_classes, feature_extract, use_pretrained=False)
    model_ft = load_model(model_ft, device, model_file)
    feature_extract_str = 'feature_extract' if feature_extract else 'finetuning'
    image_name = f'model_{target_column_name}_predict_result_{feature_extract_str}1.png'
    visualize_model(model_ft, dataloader, class_names, image_name,
                    num_images=num_images)


def predict_category_and_condition_for_image_batch(models, test_columns, dataloader):
    for col in test_columns:
        models[col].eval()

    labels = dict(zip(test_columns, [[], []]))

    for input, _ in dataloader:
        input = input.to(device)
        for col in test_columns:
            outputs = models[col](input)
            _, preds = torch.max(outputs, 1)
            labels[col].extend(list(preds.data.cpu().numpy()))
    return labels


values_dict = {'condition': {'Fair': 0, 'Good': 1, 'Like New': 2, 'New': 3, 'Poor': 4},
               'category': {'Amplifiers & Effects': 0,
                            'Band & Orchestra': 1,
                            'Bass Guitars': 2,
                            'Brass Instruments': 3,
                            'DJ, Electronic Music & Karaoke': 4,
                            'Drums & Percussion': 5,
                            'Guitars': 6,
                            'Instrument Accessories': 7,
                            'Keyboards': 8,
                            'Live Sound & Stage': 9,
                            'Microphones & Accessories': 10,
                            'Other': 11,
                            'Stringed Instruments': 12,
                            'Studio Recording Equipment': 13,
                            'Wind & Woodwind Instruments': 14}
               }

ids_dict = {col: {i: val for val, i in values_dict[col].items()} for col in values_dict.keys()}

if __name__ == '__main__':
    model_name = 'resnet'
    num_epochs = 15
    batch_size = 8
    feature_extract = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    what_to_do = 'production_like_run'
    target_column = 'condition'
    LOAD_TRAINED_MODEL = False

    df = pd.read_excel('products.xlsx')
    df.dropna(inplace=True, subset=['category', 'condition'])

    if what_to_do == 'train_model_for_specific_column':
        get_model_for_target_column(df, 'id', target_column, model_name,
                                    num_epochs, batch_size, feature_extract, device)

    elif what_to_do == 'test_trained_model_for_specific_column':
        # tests trained model on a set number of random test images from dataset
        num_images = 8
        model_file = f'best_model_resnet50_{target_column}_finetuning.pt'
        load_and_test_trained_model(model_file, 8, 'id', target_column, model_name,
                                    batch_size, feature_extract, device)

    elif what_to_do == 'production_like_run':
        n_samples_to_test = 300
        test_columns = ['category', 'condition']
        test_sample = df[['id', 'category', 'condition']].sample(n_samples_to_test, random_state=random_state)
        # substitute  names with numbers (categorical encoding form the full dataset)
        for col in test_columns:
            test_sample[f'{col}_id'] = test_sample[col].apply(lambda x: values_dict[col][x])
        # ids of images are the same for predicting both category and condition
        # i use here dataloaders as anyway we need to process batch of images,
        # so why not utilize the structure we already have (+ it works fine with transformations)
        image_dataset = ImageData('.', 'img_n', test_sample['id'].values,
                                  test_sample['category_id'].values, data_transforms['val'])
        dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size,
                                                 shuffle=True, num_workers=4)
        models_dict = dict(zip(test_columns, [[], []]))
        for col in test_columns:
            n_classes = len(values_dict[col])
            models_dict[col], _ = initialize_model(model_name, n_classes, feature_extract, use_pretrained=False)
            models_dict[col] = load_model(models_dict[col], device, f'best_model_resnet50_{col}_finetuning.pt')

        predicted_labels = predict_category_and_condition_for_image_batch(models_dict, test_columns, dataloader)
        for col in test_columns:
            test_sample[f'{col}_pred_id'] = predicted_labels[col]
            test_sample[f'{col}_pred'] = [ids_dict[col][i] for i in predicted_labels[col]]
        test_sample.to_csv('test_result.csv', index=False)
