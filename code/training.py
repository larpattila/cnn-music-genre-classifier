
from copy import deepcopy

import numpy as np

import torch
from torch import cuda, no_grad
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam
from model import MyModel
from pathlib import Path
from data_handling import get_dataset, get_data_loader

from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

def main():

    genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
              'jazz', 'metal', 'pop', 'reggae', 'rock']

    device = 'cuda' if cuda.is_available() else 'cpu'
    print(f"Process on {device}", end="\n\n")

    epochs = 30

    model = MyModel()

    model = model.to(device)

    optimizer = Adam(params=model.parameters(), lr=1e-3, weight_decay=5e-4)

    loss_function = nn.CrossEntropyLoss()

    batch_size = 32

    data_path = Path('Data/')

    train_dataset = get_dataset('training', data_path)
    train_loader = get_data_loader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=True)

    valid_dataset = get_dataset('validation', data_path)
    valid_loader = get_data_loader(dataset=valid_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=True)

    test_dataset = get_dataset('testing', data_path)
    test_loader = get_data_loader(dataset=test_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True)

    # Early stopping variables
    lowest_validation_loss = 1e10
    best_validation_epoch = 0
    patience = 5
    patience_counter = 0
    best_model = None

    for epoch in range(epochs):
        epoch_loss_training = []
        epoch_loss_validation = []

        model.train()
        for batch in train_loader:

            optimizer.zero_grad()

            x, y = batch

            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)

            loss = loss_function(input=y_hat, target=y)

            loss.backward()

            optimizer.step()

            epoch_loss_training.append(loss.item())

        model.eval()

        with no_grad():

            for batch in valid_loader:

                x_val, y_val = batch
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                y_hat = model(x_val)

                loss = loss_function(input=y_hat, target=y_val)

                epoch_loss_validation.append(loss.item())

        epoch_loss_validation = np.array(epoch_loss_validation).mean()
        epoch_loss_training = np.array(epoch_loss_training).mean()

        if epoch_loss_validation < lowest_validation_loss:
            lowest_validation_loss = epoch_loss_validation
            patience_counter = 0
            best_model = deepcopy(model.state_dict())
            best_validation_epoch = epoch
        else:
            patience_counter += 1

        if patience_counter >= patience or epoch == epochs - 1:
            print('\nExiting due to early stopping', end='\n\n')
            print(f'Best epoch {best_validation_epoch} with loss {lowest_validation_loss}', end='\n\n')
            if best_model is None:
                print('No best model. ')
            else:
                torch.save(best_model, 'best_model_hrtf.pt')
                model.load_state_dict(best_model)

                print('Starting testing', end=' | ')
                testing_loss = []
                labels_test = []
                predictions_test = []
                model.eval()
                with no_grad():
                    for batch in test_loader:

                        x_test, y_test = batch
                        x_test = x_test.to(device)
                        y_test = y_test.to(device)

                        y_hat = model(x_test)

                        loss = loss_function(input= y_hat, target= y_test)
                        testing_loss.append(loss.item())

                        predictions_test.append(y_hat.argmax(dim=-1).cpu().numpy())
                        labels_test.append(y_test.cpu().numpy())

                testing_loss = np.array(testing_loss).mean()
                print(f'Testing loss: {testing_loss:7.4f}')

                predictions_test = np.concatenate(predictions_test)
                labels_test = np.concatenate(labels_test)
                accuracy = accuracy_score(labels_test, predictions_test)
                print(f'Testing accuracy: {accuracy:7.4f}')

                confusion_matrix_test = confusion_matrix(labels_test,
                                                         predictions_test)
                cm_display = ConfusionMatrixDisplay(confusion_matrix= confusion_matrix_test,
                                                    display_labels= genres)
                cm_display.plot()
                plt.show()

                break
        print(f'Epoch: {epoch:03d} | '
              f'Mean training loss: {epoch_loss_training:7.4f} | '
              f'Mean validation loss {epoch_loss_validation:7.4f}')

if __name__ == '__main__':
    main()



#Exiting due to early stopping

#Best epoch 7 with loss 1.7016577317434198

#Starting testing | Testing loss:  1.6006
#Testing accuracy:  0.4827


















