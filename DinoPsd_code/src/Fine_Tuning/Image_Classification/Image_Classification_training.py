import torch
from torch import nn
import numpy as np
from tqdm.notebook import tqdm

from setup import model, device 

from Fine_Tuning.Neuro_Classification.Neuro_Classification_dataset import training_loader, test_loader
from Fine_Tuning.Neuro_Classification.Neuro_Classification_Head import head


complete_model = nn.Sequential(model, head)
complete_model.eval()

ft_optimizer = torch.optim.Adam(complete_model.parameters(), lr=3e-4)
complete_model.to(device)
ft_loss_fn = nn.BCELoss()

def image_classification_training(epochs):
    complete_model.train()
    loss_list = []
    prediction_list = []
    test_accuracies = []
    for _ in tqdm(range(epochs), desc=f'Epoch:'):
        epoch_loss_list = []
        for images, one_hot_gts in tqdm(training_loader, desc='Training', leave=False):
            images = images.to(torch.float32).to(device)
            
            output = complete_model(images).to(torch.float64)
            
            gt = one_hot_gts
            gt = gt.to(device)
            loss=0
            for out, true in zip(output,gt):
                loss += ft_loss_fn(out,true)
                
            loss.backward()
            ft_optimizer.step()
            ft_optimizer.zero_grad()
            
            epoch_loss_list.append(loss.detach().cpu().numpy())

        loss_list.append(np.mean(epoch_loss_list))

        complete_model.eval()
        with torch.no_grad():
            score = 0
            total = 0
            for images, one_hot_gts in tqdm(test_loader, desc='Testing', leave=False):
                
                images = images.to(torch.float32).to(device)
                outputs = complete_model(images) # shape (batch_size, nb_classes)
                
                for output, one_hot_gt in zip(outputs, one_hot_gts):
                    predicted_idx = torch.argmax(output).item()
                    true_idx = torch.argmax(one_hot_gt).item()
                    prediction_list.append([predicted_idx, true_idx])
                    
                    if predicted_idx == true_idx:
                        score += 1
                    total += 1
                batch_score = 100*score/total
            test_accuracies.append(batch_score)

    return loss_list, prediction_list, test_accuracies