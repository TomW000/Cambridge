from utils import os, transforms

#path = os.path.join(os.getcwd(), 'neurotransmitter_data') 
path = '/Users/tomw/Documents/MVA/Internship/Cambridge/Samia/Synister_NeurotransmitterPrediction/neurotransmitter_data'
all_data = []
for date in os.listdir(path):
    if date != '.DS_Store':
        for types in os.listdir(os.path.join(path, date)):
            if types != '.DS_Store':
                for file in os.listdir(os.path.join(path, date, types)):
                    all_data.append([os.path.join(path, date, types, file), types])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])