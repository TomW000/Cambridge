import torch 
from torch import nn
from tqdm.notebook import tqdm
import numpy as np

from .Psd_Prediction_Head import detection_head
from .Psd_Prediction_dataset import training_loader, test_loader
from setup import device

optimizer = torch.optim.Adam(detection_head.parameters(), lr=3e-4)
loss_fn = nn.BCELoss()


def psd_detection_training(epochs):
    detection_head.train()
    loss_list = []
    prediction_list = []
    test_accuracies = []
    for _ in tqdm(range(epochs), desc=f'Epoch:'):
        epoch_loss_list = []
        proportion_list = []
        for embeddings, one_hots in tqdm(training_loader, desc='Training', leave=False):
            embeddings = embeddings.to(device)
            outputs = detection_head(embeddings).to(torch.float64)
            
            one_hots = one_hots.to(device)
            loss=0
            for output, gt in zip(outputs, one_hots):
                loss += loss_fn(output,gt)
                
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss_list.append(loss.detach().cpu().numpy())
            
            proportion_list.append(one_hots)
                
        loss_list.append(np.mean(epoch_loss_list))

        detection_head.eval()
        with torch.no_grad():
            score = 0
            total = 0
            for embeddings, one_hots in tqdm(test_loader, desc='Testing', leave=False):
                embeddings = embeddings.to(device)
                outputs = detection_head(embeddings) # shape (batch_size, nb_classes)
                
                for output, gt in zip(outputs, one_hots):
                    predicted_idx = torch.argmax(output).item()
                    true_idx = torch.argmax(gt).item()
                    prediction_list.append([predicted_idx, true_idx])
                    
                    if predicted_idx == true_idx:
                        score += 1
                    total += 1
                batch_score = 100*score/total
            test_accuracies.append(batch_score)

    return loss_list, proportion_list, prediction_list, test_accuracies