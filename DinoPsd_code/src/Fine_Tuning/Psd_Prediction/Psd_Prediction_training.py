import torch 
from torch import nn
from tqdm.notebook import tqdm
import numpy as np
import os 

from .Psd_Prediction_Head import detection_head
from .Psd_Prediction_dataset import training_loader, test_loader
from setup import device, model_weights_path, model_size, resize_size

optimizer = torch.optim.Adam(detection_head.parameters(), lr=3e-4)
loss_fn = nn.BCELoss()

def psd_detection_training(epochs):
    detection_head.train()
    loss_list = []
    prediction_list = []
    test_accuracies = []

    for _ in tqdm(range(epochs), desc='Epochs'):
        epoch_loss_list = []

        for embeddings, one_hots in tqdm(training_loader, desc='Training', leave=False):
            embeddings = embeddings.to(device)
            one_hots = one_hots.to(device).float()  # Make sure targets are float for BCELoss

            optimizer.zero_grad()
            outputs = detection_head(embeddings).float()  # Ensure float32

            loss = loss_fn(outputs, one_hots)
            loss.backward()
            optimizer.step()

            epoch_loss_list.append(loss.item())

        loss_list.append(np.mean(epoch_loss_list))

        # Evaluation phase
        detection_head.eval()
        with torch.no_grad():
            score = 0
            total = 0
            for embeddings, one_hots in tqdm(test_loader, desc='Testing', leave=False):
                embeddings = embeddings.to(device)
                one_hots = one_hots.to(device)

                outputs = detection_head(embeddings)

                for output, gt in zip(outputs, one_hots):
                    predicted_idx = torch.argmax(output).item()
                    true_idx = torch.argmax(gt).item()
                    prediction_list.append([predicted_idx, true_idx])

                    if predicted_idx == true_idx:
                        score += 1
                    total += 1

            batch_score = 100 * score / total if total > 0 else 0
            test_accuracies.append(batch_score)

        detection_head.train()

    # Save model
    torch.save(detection_head.state_dict(), os.path.join(
        model_weights_path, f'psd_detection_head_{model_size}_{resize_size}.pt'))

    return loss_list, prediction_list, test_accuracies
