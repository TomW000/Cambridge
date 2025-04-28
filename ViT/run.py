from dataset import ViTDataset
import config

datamodule = ViTDataset(
    data_path=config['data_param']['data_path'],
    dataset_proportions=config['data_param']['dataset_proportions'],
    batch_sizes=config['data_param']['batch_sizes'],
    nb_workers=config['data_param']['nb_workers']
    )