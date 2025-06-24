from setup import umap, torch, tqdm, np, px, PCA, KMeans, device, model, feat_dim, resize_size
from napari_dinosim.utils import get_img_processing_f
from perso_utils import get_fnames, load_image
from napari_dinosim.dinoSim_pipeline import crop_data_with_overlap
from napari_dinosim.utils import resizeLongestSide, mirror_border
from DINOSim import *
from setup import Union


class DINOPsd:
    def __init__(self, 
                 model,
                 device,
                 patch_size = (14,14)
                 ):
        self.model = model
        self.device = device
        self.patch_size = patch_size
        
    def resize_image(patch_size, 
                    image: Union[np.array, torch.tensor]
                    ):
        w,h,c = image.shape
        #compatible = (np.divide([w,h],[14,14])=1)
        
        
    def set_reference_vectors(images: list[Union[np.array, torch.tensor]],
                              labels: list[str], 
                              coordinate: list[tuple]
                              ):
        pass