from common import os, Image, torch, T 

def data_loader():
    train_x, train_y = [], []
    test_x, test_y = [], []

    transform = T.Compose([T.PILToTensor(),
                           T.ConvertImageDtype(torch.float32)])

    data_dir = os.path.join(os.getcwd(), 'EM_Data', 'original')

    data_sets = [f for f in os.listdir(data_dir) if not f.startswith('.')]

    for data_set in data_sets:
        dataset_path = os.path.join(data_dir, data_set)
        
        if data_set == 'train':
            for coordinate in os.listdir(dataset_path):
                coord_path = os.path.join(dataset_path, coordinate)
                if not os.path.isdir(coord_path):
                    continue
                    
                if coordinate == 'x':
                    for image in os.listdir(coord_path):
                        if image.startswith('.'):
                            continue
                        img_path = os.path.join(coord_path, image)
                        with Image.open(img_path) as img:
                            train_x.append(transform(img))
                else:
                    for image in os.listdir(coord_path):
                        if image.startswith('.'):
                            continue
                        img_path = os.path.join(coord_path, image)
                        with Image.open(img_path) as img:
                            train_y.append(transform(img))
        else:
            for coordinate in os.listdir(dataset_path):
                coord_path = os.path.join(dataset_path, coordinate)
                if not os.path.isdir(coord_path):
                    continue
                    
                if coordinate == 'x':
                    for image in os.listdir(coord_path):
                        if image.startswith('.'):
                            continue
                        img_path = os.path.join(coord_path, image)
                        with Image.open(img_path) as img:
                            test_x.append(transform(img))
                else:
                    for image in os.listdir(coord_path):
                        if image.startswith('.'):
                            continue
                        img_path = os.path.join(coord_path, image)
                        with Image.open(img_path) as img:
                            test_y.append(transform(img))

    return train_x, train_y, test_x, test_y