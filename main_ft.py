from datetime import datetime
from collections import Counter

import os
import hydra
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
from models import Resnet, ElderNet, TremorNet

from dataset.transformations import RotationAxis, RandomSwitchAxis
from dataset.dataloader import FT_Dataset
from utils import EarlyStopping, load_weights

from tqdm import tqdm

import models
import utils
import constants
import postprocessing

import warnings

warnings.filterwarnings("ignore")

now = datetime.now()

TEST_SIZE = 0.125
RESAMPLED_HZ = 30
NUM_BINS_FFT = 10
NUM_FEATURES_FFT = 2
NUM_AXIS_FFT = 4  # number of axes includ the magnitude


def convert_to_binary_classification(X, Y, groups, classes_to_remove=-1):
    """
    Convert a multi-class classification problem to a binary classification problem.

    Parameters:
    X (numpy.ndarray): Input data with shape (n_samples, n_features, n_features).
    Y (numpy.ndarray): Target labels with shape (n_samples, n_classes).
    groups (numpy.ndarray): Group labels
    convert (bool): Flag to indicate whether to convert the multi-class problem to binary. Default is False.
    class_to_remove (int): Class label to remove if convert is True. Default is -1.

    Returns:
    X (numpy.ndarray): Modified input data with shape (n_samples, n_features, n_features).
    Y (numpy.ndarray): Modified target labels with shape (n_samples, 2).
    groups (numpy.ndarray): Modified group labels.
    """
    # Extract the tremor score from the target labels
    tremor_score = np.argmax(Y, axis=1)

    # Create a mask to remove samples with the specified class
    mask = np.isin(tremor_score, classes_to_remove, invert=True)
    X = X[mask]
    tremor_score = tremor_score[mask]
    groups = groups[mask]

    # Convert the tremor score to binary labels in one-hot encoding
    tremor_labels = (tremor_score > 0).astype(int)
    Y = np.zeros((len(tremor_score), 2))  # one-hot encoding
    Y[np.arange(tremor_labels.size), tremor_labels] = 1

    return X, Y, groups


def rebalance_data(data, labels, subjects, desired_distribution=None):
    """
    Rebalance the dataset so that each class has an equal number of samples or follow a desired distribution.

    Parameters:
        data (numpy.ndarray or list): Input data array.
        labels (numpy.ndarray or list): Corresponding labels for the data.
        desired_distribution (dict or None): A dictionary specifying the desired proportion of each class, 
                                             e.g., {0: 0.4, 1: 0.4, 2: 0.2}. 
                                             If None, all classes will have equal proportion.

    Returns:
        balanced_data (list): Rebalanced dataset.
        balanced_labels (list): Rebalanced labels.
    """
    # Ensure input is numpy array for ease of manipulation
    # If y_pred is in one-hot encoded format, convert it to class indices
    if len(np.array(labels).shape) > 1:
        labels_decoded = torch.argmax(torch.tensor(labels), dim=1).numpy()
    data = np.array(data)
    labels_decoded = np.array(labels_decoded)

    # Count the number of samples per class
    label_counts = Counter(labels_decoded)

    # Determine the minimum class size or calculate based on desired distribution
    if desired_distribution is None:
        # Equal proportions for each class
        min_class_size = min(label_counts.values())
        desired_class_counts = {label: min_class_size for label in label_counts}
    else:
        total_samples = len(labels_decoded)
        # Calculate the number of samples desired for each class based on the given proportions
        desired_class_counts = {label: int(total_samples * proportion) for label, proportion in desired_distribution.items()}

    # Rebalance the dataset
    balanced_data = []
    balanced_labels = []
    balanced_subjects = []

    for label, count in label_counts.items():
        # Get indices of all samples for the current label
        label_indices = np.where(labels_decoded == label)[0]

        if desired_class_counts[label] < count:
            # Undersample: Randomly select the desired number of samples for the current label
            selected_indices = np.random.choice(label_indices, size=desired_class_counts[label], replace=False)
        else:
            # Oversample: Randomly select with replacement to reach the desired number of samples
            selected_indices = np.random.choice(label_indices, size=desired_class_counts[label], replace=True)

        # Add the selected data and labels to the balanced lists
        balanced_data.extend(data[selected_indices])
        balanced_labels.extend(labels[selected_indices])
        balanced_subjects.extend(subjects[selected_indices])

    return np.array(balanced_data), np.array(balanced_labels), np.array(balanced_subjects)


def check_performance_post(labels, predictions):
    """Check the performance of the model after the post-processing stage.

           Parameters:
           - labels: Numpy array of shape (n_points)
           - predictions: Numpy array of shape (n_points)

           Returns:
           - accuracy, specificity, recall, precision, F1Score
           """
    tp = np.sum(((predictions == labels) & (labels == 1)))
    tn = np.sum(((predictions == labels) & (labels == 0)))
    fp = np.sum(((predictions != labels) & (labels == 0)))
    fn = np.sum(((predictions != labels) & (labels == 1)))

    accuracy = 100 * ((tp + tn) / (tp + tn + fp + fn))
    specificity = 100 * (tn / (1 + tn + fp))
    recall = 100 * ((1 + tp) / (1 + tp + fn))
    precision = 100 * ((1 + tp) / (1 + tp + fp))
    f1score = 2 * (precision * recall) / (precision + recall)

    return np.round(accuracy, 2), np.round(specificity, 2), np.round(recall, 2), np.round(precision, 2), np.round(
        f1score, 2)


def predict(model, data_loader, device):
    """
    Iterate over the dataloader and do prediction with a pytorch model.

    :param nn.Module model: pytorch Module
    :param DataLoader data_loader: pytorch dataloader
    :param device: pytorch map device

    :return: true labels, model predictions, pids
    :rtype: (np.ndarray, np.ndarray, np.ndarray)
    """

    predictions_list = []
    true_list = []
    model.eval()

    for i, (x, y) in enumerate(tqdm(data_loader, mininterval=60)):
        with torch.inference_mode():
            x = x.to(device, dtype=torch.float)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            preds = probs[:, 1]  # gait probabilities
            y = torch.argmax(y, dim=1)
            true_list.append(y)
            predictions_list.append(preds.cpu())

    true_list = torch.cat(true_list)
    predictions_list = torch.cat(predictions_list)

    return true_list.numpy(), predictions_list.numpy()


def evaluate_model(model, val_loader, device, loss_fn):
    model.eval()
    losses = []
    acces = []
    for i, (x, y) in enumerate(val_loader):
        with torch.no_grad():
            x = x.to(device, dtype=torch.float)
            true_y = y.to(device, dtype=torch.long)

            logits = model(x)
            # loss = loss_fn(logits.float(), true_y.float())
            loss = loss_fn(logits.float(), torch.argmax(true_y, dim=1))

            probs = F.softmax(logits, dim=1)
            pred_y = torch.argmax(probs, dim=1)

            val_acc = torch.sum(pred_y == torch.argmax(true_y, dim=1)).float()
            val_acc /= pred_y.size(0)  # Normalization to get a probability like value

            losses.append(loss.cpu().detach())
            acces.append(val_acc.cpu().detach())
    losses = np.array(losses)
    acces = np.array(acces)
    return np.mean(losses), np.mean(acces)


def shufflegroupkfold(x, y, groups, n_splits=5):
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    labels = np.argmax(y, axis=1)
    for i, (train_index, test_index) in enumerate(sgkf.split(x, labels, groups)):
        print(f"Fold {i}:")
        yield train_index, test_index


def set_seed(device, my_seed=0):
    random_seed = my_seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(random_seed)


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


@hydra.main(config_path="conf", config_name="config_ft", version_base='1.1')
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = cfg.model.num_epochs
    lr = cfg.model.lr
    # dense_labeling = cfg.data.dense_labeling
    batch_size = cfg.model.batch_size
    # window_size = cfg.dataloader.epoch_len * cfg.dataloader.sample_rate

    augmentations = []
    if cfg.augmentation.axis_switch:
        augmentations.append(RandomSwitchAxis())

    if cfg.augmentation.rotation:
        augmentations.append(RotationAxis())

    my_transform = transforms.Compose(augmentations) if augmentations else None

    main_log_dir = cfg.data.log_path
    dt_string = now.strftime("%d-%m-%Y_%H_%M_%S")
    run_name = cfg.model.name + "FT_lr_" \
               + str(lr) \
               + "_batchSize_" \
               + str(batch_size) \
               + '_n_epochs_' \
               + str(num_epochs) \
               + dt_string

    output_path = os.path.join(
        PROJECT_ROOT,
        main_log_dir,
        "models",
        run_name
    )

    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    # Path to save the model weights
    weights_path = os.path.join(output_path, 'weights.pt')

    # Load the data (resample to 30 hz and divided into windows of 10 sec)
    X = pickle.load(open(os.path.join(PROJECT_ROOT, cfg.data.data_root, 'WindowsData.p'), 'rb'))  # (n_windows, 3, 300)
    # fft_product = pickle.load(open(os.path.join(PROJECT_ROOT, cfg.data.data_root, 'WindowsFFT.p'), 'rb'))  # (n_windows, 60)
    # fft_product_expanded = np.expand_dims(fft_product, axis=1)  # Shape: (n_windows, 1, 60)
    # fft_product_expanded = np.tile(fft_product_expanded, (1, X.shape[1], 1))  # Shape: (n_windows, 3, 60)
    # X = np.concatenate((X, fft_product_expanded), axis=-1)  # Shape: (n_windows, 3, 360)
    Y = pickle.load(open(os.path.join(PROJECT_ROOT, cfg.data.data_root, 'WindowsLabels.p'), 'rb'))  # (n_windows,#classes)
    # The groups vector indicates the subject_id of each window, which is needed for subject-wise division.
    groups = pickle.load(open(os.path.join(PROJECT_ROOT, cfg.data.data_root, 'WindowsSubjects.p'), 'rb'))  # (n_windows,)

    if not cfg.model.multiclass:
        X, Y, groups = convert_to_binary_classification(X, Y, groups, classes_to_remove=cfg.model.classes_to_remove)
    if cfg.model.rebalance:
        X, Y, groups = rebalance_data(X, Y, groups)

    seeds = constants.SEEDS
    # List of performance metrics names
    performance_metrics_names = ['accuracy', 'specificity', 'recall', 'precision', 'F1score']
    # Initialize a dictionary to store the arrays
    performance_dict = {metric_name: np.zeros(len(seeds)) for metric_name in performance_metrics_names}
    #
    curves_arrays = {
        'roc': {},
        'pr': {},
        'performance': {}
    }
    for seed in range(len(seeds)):
        set_seed(my_seed=seeds[seed], device=device)
        labels = []
        predictions = []
        for train_idxs, test_idxs in shufflegroupkfold(X, Y, groups):
            X_train, Y_train, groups_train = X[train_idxs], Y[train_idxs], groups[train_idxs]
            X_test, Y_test, groups_test = X[test_idxs], Y[test_idxs], groups[test_idxs]

            # prepare training and validation sets
            folds = GroupShuffleSplit(
                1, test_size=TEST_SIZE, random_state=41
            ).split(X_train, Y_train, groups=groups_train)
            train_idx, val_idx = next(folds)

            x_train = X_train[train_idx]
            x_val = X_train[val_idx]

            y_train = Y_train[train_idx]
            y_val = Y_train[val_idx]

            train_dataset = FT_Dataset(x_train,
                                       y_train,
                                       name="training",
                                       cfg=cfg,
                                       transform=my_transform)
            val_dataset = FT_Dataset(x_val,
                                     y_val,
                                     name="validation",
                                     cfg=cfg,
                                     transform=my_transform)

            test_dataset = FT_Dataset(X_test,
                                      Y_test,
                                      name="prediction",
                                      cfg=cfg,
                                      transform=my_transform)

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False
            )

            # balancing to 90% notwalk, 10% walk
            # walk = np.sum(y_train[:, 1])
            # notwalk = np.sum(y_train[:, 0])
            # class_weights = [(walk * 9.0) / notwalk, 1.0]
            # [0 tremor = 1/0.68, 1 tremor = 1/0.22, 3, 4, 5]
            class_weights = [len(y_train) / sum(y_train[:, i]) for i in range(len(y_train[0]))]
            # class_weights = [1 for _ in range(len(y_train[0]))]
            print(f"[*] DEBUG: CLASS WEIGHTS = {class_weights}")
            # if cfg.model.multiclass:
            # else:
            #     class_weights =
            #############################################
            # Set the Model
            #############################################
            # Instantiate a network architecture every fold (to prevent weight leakage between folds)
            if cfg.model.net == 'ElderNet':
                feature_extractor = Resnet().feature_extractor
                model = getattr(models, cfg.model.net)(feature_extractor, head=cfg.model.head,
                                                       non_linearity=cfg.model.non_linearity, is_eva=True)
            elif cfg.model.net == 'TremorNet':
                feature_extractor = Resnet().feature_extractor
                model = getattr(models, cfg.model.net)(feature_extractor)
            else:
                model = getattr(models, cfg.model.net)(is_eva=True)

            # Choose if to use an already pretrained model (such as the UKB model)
            print('model initalizing...')
            if cfg.model.pretrained:
                # Use the model from the UKB paper
                if not cfg.model.ssl_checkpoint_available:
                    pretrained_model = utils.get_sslnet(pretrained=True)
                    feature_extractor = pretrained_model.feature_extractor
                    model = Resnet(output_size=len(class_weights), feature_extractor=feature_extractor, is_eva=True)
                    if cfg.model.net == 'ElderNet':
                        model = ElderNet(feature_extractor, cfg.model.head, output_size=len(class_weights), is_eva=True)
                    elif cfg.model.net == 'TremorNet':
                        model = TremorNet(feature_extractor, input_size=1024,
                                          fft_size=NUM_FEATURES_FFT * NUM_AXIS_FFT * NUM_BINS_FFT,
                                          output_size=len(class_weights), num_layers=1)
                # Use a pretrained model of your own
                else:
                    print(f"LOADING FROM {cfg.model.trained_model_path}")
                    load_weights(cfg.model.trained_model_path, model, device)

            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)

            # Define the loss function
            if class_weights is not None:
                class_weights = torch.FloatTensor(class_weights).to(device)
                loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            else:
                loss_fn = nn.CrossEntropyLoss()

            early_stopping = EarlyStopping(patience=cfg.model.patience, path=weights_path, verbose=True)
            print('Training SSL')
            for epoch in range(num_epochs):
                model.train()
                train_losses = []
                train_acces = []
                for i, (x, y) in enumerate(tqdm(train_loader)):
                    x.requires_grad_(True)
                    x = x.to(device, dtype=torch.float)
                    true_y = y.to(device, dtype=torch.long)

                    optimizer.zero_grad()

                    logits = model(x)
                    # loss = loss_fn(logits.float(), true_y.float())
                    loss = loss_fn(logits.float(), torch.argmax(true_y, dim=1))
                    loss.backward()
                    optimizer.step()

                    # Convert logits to probabilities using softmax activation function
                    probs = F.softmax(logits, dim=1)
                    # Extract the gait probabilities
                    pred_y = torch.argmax(probs, dim=1)
                    # train_acc = torch.sum(pred_y == true_y[:, 1])
                    train_acc = torch.sum(pred_y == torch.argmax(true_y, dim=1))
                    train_acc = train_acc / (pred_y.size()[0])

                    train_losses.append(loss.cpu().detach())
                    train_acces.append(train_acc.cpu().detach())

                val_loss, val_acc = evaluate_model(model, val_loader, device, loss_fn)

                epoch_len = len(str(num_epochs))
                print_msg = (
                        f"[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] | "
                        + f"train_loss: {np.mean(train_losses):.3f} | "
                        + f"train_acc: {np.mean(train_acces):.3f} | "
                        + f"val_loss: {val_loss:.3f} | "
                        + f"val_acc: {val_acc:.2f}"
                )

                print(print_msg)
                early_stopping(val_loss, model)
                if early_stopping.early_stop:
                    print('Early stopping')
                    print(f'SSLNet weights saved to {weights_path}')
                    break

            # Predict the tset fold
            Y_test_true, Y_test_pred = predict(model, test_loader, device)
            # Append the predictions of the current test fold
            labels.append(Y_test_true)
            predictions.append(Y_test_pred)

        # labels = np.concatenate(labels)
        # predictions = np.concatenate(predictions)
        # # Calculate the ROC curve
        # fpr, tpr, thresholds = roc_curve(labels, predictions)
        # roc_auc = roc_auc_score(labels, predictions)  # area under the curve
        # curves_arrays['roc'][seed] = (fpr, tpr, roc_auc)
        # # Calculate the PR curve
        # precision, recall, thresholds = precision_recall_curve(labels, predictions)
        # auprc = average_precision_score(labels, predictions)  # area under the curve
        # curves_arrays['pr'][seed] = (recall, precision, auprc)
        # # Calculate the recall-precision-f1score curve
        # eps = 1e-15
        # f1 = (2 * precision * recall) / (precision + recall + eps)
        # intersection = np.where(precision > recall)[0][0]
        # curves_arrays['performance'][seed] = (thresholds, precision, recall, f1, intersection)

                # Initialize dictionaries to store results
        curves_arrays = {'roc': {}, 'pr': {}, 'performance': {}}
        performance_dict = {'accuracy': {}, 'specificity': {}, 'recall': {}, 'precision': {}, 'F1score': {}}
        seed = 0  # Replace with your specific seed or identifier for your experiment

        # Number of classes
        n_classes = predictions.shape[1]

        # Iterate over each class
        for i in range(n_classes):
            # Binarize the labels for the current class
            y_true = (labels == i).astype(int)
            y_score = predictions[:, i]

            # Calculate ROC curve and AUC for the current class
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            roc_auc = roc_auc_score(y_true, y_score)
            curves_arrays['roc'][i] = (fpr, tpr, roc_auc)

            # Calculate Precision-Recall curve and AUC for the current class
            precision, recall, thresholds_pr = precision_recall_curve(y_true, y_score)
            auprc = average_precision_score(y_true, y_score)
            curves_arrays['pr'][i] = (recall, precision, auprc)

            # Calculate F1 Score, Precision-Recall Curve metrics
            eps = 1e-15
            f1 = (2 * precision * recall) / (precision + recall + eps)
            intersection = np.where(precision > recall)[0][0] if len(precision) > 0 and len(recall) > 0 else None
            curves_arrays['performance'][i] = (thresholds_pr, precision, recall, f1, intersection)


        # Round predictions
        final_preds = np.round(predictions)

        # Compute the performance's metrics after post-processing
        labels = labels.astype(final_preds.dtype)  # for reliable comparison
        performance_dict['accuracy'][seed], performance_dict['specificity'][seed], performance_dict['recall'][seed], \
            performance_dict['precision'][seed], performance_dict['F1score'][seed] = \
            check_performance_post(labels, final_preds)

    with open(os.path.join(output_path, 'performance_matrix.p'), 'wb') as OutputFile:
        pickle.dump(performance_dict, OutputFile)

    # Plot PR and ROC curves for all seeds
    postprocessing.plot_curves_for_seeds(seeds, curves_arrays, output_path, 'pr')
    postprocessing.plot_curves_for_seeds(seeds, curves_arrays, output_path, 'roc')
    postprocessing.plot_curves_for_seeds(seeds, curves_arrays, output_path, 'performance')


if __name__ == '__main__':
    main()
