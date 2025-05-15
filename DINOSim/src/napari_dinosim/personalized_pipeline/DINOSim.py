from setup import umap, torch, tqdm, np, px, PCA, KMeans, device, model, feat_dim, resize_size
from napari_dinosim.utils import get_img_processing_f
from perso_utils import get_fnames, load_image
from napari_dinosim.dinoSim_pipeline import crop_data_with_overlap
from napari_dinosim.utils import resizeLongestSide, mirror_border


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------


class DinoSim_pipeline:
    """A pipeline for computing and managing DINOSim.

    This class handles the computation of DINOSim, using DINOv2 embeddings, manages reference
    vectors, and computes similarity distances. It supports processing large images through
    a sliding window approach with overlap.

    Args:
        model: The DINOv2 model to use for computing embeddings
        model_patch_size (int): Size of patches used by the DINOv2 model
        device: The torch device to run computations on (CPU or GPU)
        img_preprocessing: Function to preprocess images before computing embeddings
        feat_dim (int): Number of features of the embeddings
        dino_image_size (int, optional): Size of the image to be fed to DINOv2. Images will be resized to that size before computing them with DINOv2. Defaults to 518.
    """

    def __init__(
        self,
        model,
        model_patch_size,
        device,
        img_preprocessing,
        feat_dim,
        dino_image_size=518,
    ):
        self.model = model
        self.dino_image_size = dino_image_size
        self.patch_h = self.patch_w = self.embedding_size = (
            dino_image_size // model_patch_size
        )
        self.img_preprocessing = img_preprocessing
        self.device = device
        self.feat_dim = feat_dim

        self.reference_color = torch.zeros(feat_dim, device=device)
        self.reference_emb = torch.zeros(
            (self.embedding_size * self.embedding_size, feat_dim),
            device=device,
        )
        self.exist_reference = False

        self.embeddings = np.array([])
        self.emb_precomputed = False
        self.original_size = []
        self.overlap = (0.5, 0.5)
        self.padding = (0, 0)
        self.crop_shape = (518, 518, 1)
        self.resized_ds_size, self.resize_pad_ds_size = [], []

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def pre_compute_embeddings(
        self,
        dataset,
        overlap=(0.5, 0.5),
        padding=(0, 0),
        crop_shape=(512, 512, 1),
        verbose=True,
        batch_size=1,
    ):
        """Pre-compute DINO embeddings for the entire dataset.

        The dataset is processed in crops with optional overlap. Large images are handled
        through a sliding window approach, and small images are resized.

        Args:
            dataset: Input image dataset with shape (batch, height, width, channels)
            overlap (tuple, optional): Overlap fraction (y, x) between crops. Defaults to (0.5, 0.5).
            padding (tuple, optional): Padding size (y, x) for crops. Defaults to (0, 0).
            crop_shape (tuple, optional): Size of crops (height, width, channels). Defaults to (512, 512, 1).
            verbose (bool, optional): Whether to show progress bar. Defaults to True.
            batch_size (int, optional): Batch size for processing. Defaults to 1.
        """
        print("Precomputing embeddings")
        self.original_size = dataset.shape
        self.overlap = overlap
        self.padding = padding
        self.crop_shape = crop_shape
        b, h, w, c = dataset.shape
        self.resized_ds_size, self.resize_pad_ds_size = [], []

        # if both image resolutions are smaller than the patch size,
        # resize until the largest side fits the patch size
        if h < crop_shape[0] and w < crop_shape[0]:
            dataset = np.array(
                [
                    resizeLongestSide(np_image, crop_shape[0])
                    for np_image in dataset
                ]
            )
            if len(dataset.shape) == 3:
                dataset = dataset[..., np.newaxis]
            self.resized_ds_size = dataset.shape

        # yet if one of the image resolutions is smaller than the patch size,
        # add mirror padding until smaller side fits the patch size
        if (
            dataset.shape[1] % crop_shape[0] != 0
            or dataset.shape[2] % crop_shape[1] != 0
        ):
            desired_h, desired_w = (
                np.ceil(dataset.shape[1] / crop_shape[0]) * crop_shape[0],
                np.ceil(dataset.shape[2] / crop_shape[1]) * crop_shape[1],
            )
            dataset = np.array(
                [
                    mirror_border(
                        np_image, sizeH=int(desired_h), sizeW=int(desired_w)
                    )
                    for np_image in dataset
                ]
            )
            self.resize_pad_ds_size = dataset.shape

        # needed format: b,h,w,c
        windows = crop_data_with_overlap(
            dataset,
            crop_shape=crop_shape,
            overlap=overlap,
            padding=padding,
            verbose=False,
        )
        windows = torch.tensor(windows, device=self.device)
        windows = self._quantile_normalization(windows.float())
        prep_windows = self.img_preprocessing(windows)

        self.delete_precomputed_embeddings()
        self.embeddings = torch.zeros(
            (len(windows), self.patch_h, self.patch_w, self.feat_dim),
            device=self.device,
        )

        following_f = tqdm if verbose else lambda aux: aux
        for i in following_f(range(0, len(prep_windows), batch_size)):
            batch = prep_windows[i : i + batch_size]
            b, h, w, c = batch.shape  # b,h,w,c
            crop_h, crop_w, _ = crop_shape
            overlap = (
                overlap[0] if w > crop_w else 0,
                overlap[1] if h > crop_h else 0,
            )

            with torch.no_grad():
                if self.model is None:
                    raise ValueError("Model is not initialized")
                encoded_window = self.model.forward_features(batch)[
                    "x_norm_patchtokens"
                ]
            self.embeddings[i : i + batch_size] = encoded_window.reshape(
                encoded_window.shape[0],
                self.patch_h,
                self.patch_w,
                self.feat_dim,
            )  # use all dims

        self.emb_precomputed = True

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def _quantile_normalization(
        self,
        tensor,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
    ):
        """Normalize tensor values between quantile bounds.

        Args:
            tensor (torch.Tensor): Input tensor to normalize
            lower_quantile (float): Lower quantile bound (0-1)
            upper_quantile (float): Upper quantile bound (0-1)

        Returns:
            torch.Tensor: Normalized tensor with values between 0 and 1
        """
        flat_tensor = tensor.flatten()
        bounds = torch.quantile(
            flat_tensor,
            torch.tensor(
                [lower_quantile, upper_quantile], device=tensor.device
            ),
        )
        lower_bound, upper_bound = bounds[0], bounds[1]

        clipped_tensor = torch.clamp(tensor, lower_bound, upper_bound)
        normalized_tensor = (clipped_tensor - lower_bound) / (
            upper_bound - lower_bound + 1e-8
        )
        return normalized_tensor

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def delete_precomputed_embeddings(
        self,
    ):
        """Delete precomputed embeddings to free memory."""
        del self.embeddings
        self.embeddings = np.array([])
        self.emb_precomputed = False
        torch.cuda.empty_cache()

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def delete_references(
        self,
    ):
        """Delete reference vectors to free memory."""
        del self.reference_color, self.reference_emb, self.exist_reference
        self.reference_color = torch.zeros(self.feat_dim, device=self.device)
        self.reference_emb = torch.zeros(
            (self.embedding_size * self.embedding_size, self.feat_dim),
            device=self.device,
        )
        self.exist_reference = False
        torch.cuda.empty_cache()

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def set_reference_vector(self, list_coords, filter=None):
        """Set reference vectors from a list of coordinates in the original image space.

        Computes mean embeddings from the specified coordinates to use as reference vectors
        for similarity computation.

        Args:
            list_coords: List of tuples (batch_idx, z, x, y) specifying reference points
            filter: Optional filter to apply to the generated pseudolabels
        """
        self.delete_references()
        if len(self.resize_pad_ds_size) > 0:
            b, h, w, c = self.resize_pad_ds_size
            if len(self.resized_ds_size) > 0:
                original_resized_h, original_resized_w = self.resized_ds_size[
                    1:3
                ]
            else:
                original_resized_h, original_resized_w = self.original_size[
                    1:3
                ]
        elif len(self.resized_ds_size) > 0:
            b, h, w, c = self.resized_ds_size
            original_resized_h, original_resized_w = h, w
        else:
            b, h, w, c = self.original_size
            original_resized_h, original_resized_w = h, w

        n_windows_h = int(np.ceil(h / self.crop_shape[0]))
        n_windows_w = int(np.ceil(w / self.crop_shape[1]))

        # Calculate actual scaling factors
        scale_x = original_resized_w / self.original_size[2]
        scale_y = original_resized_h / self.original_size[1]

        # Calculate padding
        pad_left = (w - original_resized_w) / 2
        pad_top = (h - original_resized_h) / 2

        list_ref_colors, list_ref_embeddings = [], []
        for n, x, y in list_coords:
            # Apply scaling and padding to coordinates
            x_transformed = x * scale_x + pad_left
            y_transformed = y * scale_y + pad_top

            # Calculate crop index and relative position within crop
            n_crop = int(
                np.floor(x_transformed / self.crop_shape[1])
                + np.floor(y_transformed / self.crop_shape[0]) * n_windows_w
            )
            x_coord = (x_transformed % self.crop_shape[1]) / self.crop_shape[1]
            y_coord = (y_transformed % self.crop_shape[0]) / self.crop_shape[0]

            emb_id = int(n_crop + n * n_windows_h * n_windows_w)

            # Validate embedding index
            if emb_id >= len(self.embeddings):
                raise ValueError(
                    f"Invalid embedding index {emb_id} for coordinates ({n}, {x}, {y})"
                )

            x_coord = min(
                round(x_coord * self.embedding_size), self.embedding_size - 1
            )
            y_coord = min(
                round(y_coord * self.embedding_size), self.embedding_size - 1
            )

            list_ref_colors.append(self.embeddings[emb_id][y_coord, x_coord])
            list_ref_embeddings.append(self.embeddings[emb_id])

        list_ref_colors, list_ref_embeddings = torch.stack(
            list_ref_colors
        ), torch.stack(list_ref_embeddings)
        assert (
            len(list_ref_colors) > 0
        ), "No binary objects found in given masks"

        self.reference_color = torch.mean(list_ref_colors, dim=0)
        self.reference_emb = list_ref_embeddings
        self.generate_pseudolabels(filter)
        self.exist_reference = True

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def generate_pseudolabels(self, filter=None):
        """Generate pseudolabels using reference embeddings.
        
        Args:
            filter: Optional filter to apply to the generated distances
        """
        reference_embeddings = self.reference_emb.view(
            -1, self.reference_emb.shape[-1]
        )
        distances = torch.cdist(
            reference_embeddings, self.reference_color[None], p=2
        )

        if filter != None:
            distances = distances.view(
                (
                    self.reference_emb.shape[0],
                    1,
                    int(self.embedding_size),
                    int(self.embedding_size),
                )
            )
            distances = filter(distances)

        # normalize per image
        distances = self.quantile_normalization(distances)

        self.reference_pred_labels = distances.view(-1, 1)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def quantile_normalization(
        self, tensor, lower_quantile=0.01, upper_quantile=0.99
    ):
        """Normalize tensor values between quantile bounds.
        
        Args:
            tensor (torch.Tensor): Input tensor to normalize
            lower_quantile (float): Lower quantile bound (0-1)
            upper_quantile (float): Upper quantile bound (0-1)
            
        Returns:
            torch.Tensor: Normalized tensor with values between lower and upper bounds
        """
        sorted_tensor, _ = tensor.flatten().sort()
        lower_bound = sorted_tensor[
            int(lower_quantile * (len(sorted_tensor) - 1))
        ]
        upper_bound = sorted_tensor[
            int(upper_quantile * (len(sorted_tensor) - 1))
        ]

        clipped_tensor = torch.clamp(tensor, lower_bound, upper_bound)
        return (clipped_tensor - lower_bound) / (
            upper_bound - lower_bound + 1e-8
        )

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def get_d_closest_elements(self, d=0.8, normalize_distances=True, return_indices=True, verbose=False):
        """
        Get the elements in the embedding space that are close to ANY of the reference vectors.
        
        Args:
            d (float, optional): Distance threshold. Elements with distance less than d to any reference are returned. Defaults to 0.5.
            normalize_distances (bool, optional): Whether to normalize distances using quantile normalization. Defaults to True.
            return_indices (bool, optional): Whether to return the indices of the closest elements. Defaults to True.
            verbose (bool, optional): Whether to print progress information. Defaults to False.
            
        Returns:
            torch.Tensor: Tensor containing the embeddings with distance less than d to any reference vector
            torch.Tensor (optional): Indices of these embeddings if return_indices is True
            torch.Tensor (optional): Minimum distances to any reference vector if return_indices is True
        """
        if not self.exist_reference:
            raise ValueError("Reference vector not set. Call set_reference_vector first.")
        
        if not self.emb_precomputed:
            raise ValueError("Embeddings not computed. Call pre_compute_embeddings first.")
        
        if verbose:
            print("Finding elements close to reference vectors...")
            
        # Flatten all embeddings to 2D tensor: (num_embeddings*h*w, feat_dim)
        all_embeddings = self.embeddings.reshape(-1, self.feat_dim)
        
        # Flatten reference embeddings to 2D tensor: (num_ref_points*h*w, feat_dim)
        ref_embeddings = self.reference_emb.reshape(-1, self.feat_dim)
        
        # Compute distances between all embeddings and all reference embeddings
        # This will have shape (num_embeddings*h*w, num_ref_points*h*w)
        distances = torch.cdist(all_embeddings, ref_embeddings, p=2)
        
        # Get minimum distance to any reference embedding for each point
        min_distances, _ = torch.min(distances, dim=1)
        
        # Normalize distances if requested
        if normalize_distances:
            min_distances = self.quantile_normalization(min_distances.view(-1, 1)).squeeze()
        
        # Find elements with minimum distance less than the threshold
        below_threshold = min_distances < d
        closest_indices = torch.nonzero(below_threshold).squeeze()
        
        # Handle case where there's only one match
        if closest_indices.dim() == 0 and closest_indices.numel() > 0:
            closest_indices = closest_indices.unsqueeze(0)
        
        # Get the corresponding embeddings
        closest_embeddings = all_embeddings[closest_indices]
        
        # Handle case where no matches are found
        if closest_embeddings.numel() == 0:
            if verbose:
                print(f"No embeddings found within distance {d} to any reference. Try increasing the threshold.")
            return closest_embeddings
        
        if verbose:
            print(f"Found {len(closest_embeddings)} embeddings within distance {d} to reference vectors.")
            
        if return_indices:
            # Get the minimum distances for these points
            closest_distances = min_distances[closest_indices]
            
            # Convert flat indices to (batch, y, x) coordinates
            batch_size = len(self.embeddings)
            patch_size = self.patch_h * self.patch_w
            
            batch_indices = closest_indices // patch_size
            remaining_indices = closest_indices % patch_size
            y_indices = remaining_indices // self.patch_w
            x_indices = remaining_indices % self.patch_w
            
            spatial_indices = torch.stack([batch_indices, y_indices, x_indices], dim=1)
            
            return closest_embeddings, spatial_indices, closest_distances
        
        return closest_embeddings
    
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def get_k_closest_elements(self, k=10, return_indices=False):
        """
        Get the k elements in the embedding space that are closest to the reference vector.

        Args:
            k (int, optional): Number of nearest neighbors to return. Defaults to 5.
            return_indices (bool, optional): Whether to return the indices of the closest elements. Defaults to False.

        Returns:
            torch.Tensor: Tensor containing the k nearest embeddings to the reference vector
            torch.Tensor (optional): Indices of the k nearest embeddings if return_indices is True
        """
        if not self.exist_reference:
            raise ValueError("Reference vector not set. Call set_reference_vector first.")

        if not self.emb_precomputed:
            raise ValueError("Embeddings not computed. Call pre_compute_embeddings first.")

        # Flatten all embeddings to 2D tensor: (num_embeddings*h*w, feat_dim)
        all_embeddings = self.embeddings.reshape(-1, self.feat_dim)

        ref_embedings = self.reference_emb.reshape(-1, self.feat_dim)

        # Compute distances between all embeddings and the reference color vector
        distances = torch.cdist(all_embeddings, self.reference_color.unsqueeze(0), p=2)
        #distances = torch.cdist(all_embeddings, ref_embedings, p=2)

        # Get the k smallest distances and their indices
        k_smallest_values, k_smallest_indices = torch.topk(distances.squeeze(), k=k, largest=False)

        # Get the corresponding embeddings
        closest_embeddings = all_embeddings[k_smallest_indices]

        if return_indices:
            # Convert flat indices back to (batch_idx, y, x) coordinates
            batch_size = len(self.embeddings)
            patch_size = self.patch_h * self.patch_w

            batch_indices = k_smallest_indices // patch_size
            remaining_indices = k_smallest_indices % patch_size
            y_indices = remaining_indices // self.patch_w
            x_indices = remaining_indices % self.patch_w

            indices = torch.stack([batch_indices, y_indices, x_indices], dim=1)
            return closest_embeddings, indices, k_smallest_values

        return closest_embeddings

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Alias method for backwards compatibility or alternative naming convention
    def get_closest_elements(self, d=0.5, normalize_distances=True, return_indices=True, verbose=False):
        """Alias for get_elements_close_to_any_reference."""
        return self.get_elements_close_to_any_reference(d, normalize_distances, return_indices, verbose)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def diplay_features(embeddings,
                    labels,
                    include_pca=False,
                    pca_nb_components=10,
                    clustering=True,
                    nb_clusters=6,
                    nb_neighbor=5,
                    min_dist=0.01,
                    nb_components=2,
                    metric='correlation'):
    """Display features using PCA and UMAP for dimensionality reduction.
    
    Args:
        embeddings: Feature embeddings to visualize
        labels: Labels for coloring the points
        include_pca (bool): Whether to apply PCA before UMAP
        pca_nb_components (int): Number of PCA components if PCA is used
        nb_neighbor (int): Number of neighbors for UMAP
        min_dist (float): Minimum distance for UMAP
        nb_components (int): Number of UMAP components
        metric (str): Distance metric for UMAP
    """
    if len(embeddings):

        if include_pca:
            pca = PCA(n_components=pca_nb_components)
            features = pca.fit_transform(embeddings)
        else:
            features = embeddings

        reducer = umap.UMAP(
            n_neighbors=nb_neighbor,
            min_dist=min_dist,
            n_components=nb_components,
            metric=metric
            )
        embedding = reducer.fit_transform(features)
        
        if clustering:
            kmeans = KMeans(n_clusters=nb_clusters)
            kmeans.fit_transform(embedding)
            fig = px.scatter(
                x=embedding[:, 0],
                y=embedding[:, 1],
                color = kmeans.labels_,
                title = f'KMeans={clustering} ({nb_clusters}) - PCA={include_pca} ({pca_nb_components}) - UMAP (n_neighbors={nb_neighbor}, min_dist={min_dist}, n_components={nb_components}, metric={metric})',
                width=1500,
                height=1000
                )

        else:

            fig = px.scatter(
                x=embedding[:, 0],
                y=embedding[:, 1],
                color = labels,
                title = f'PCA={include_pca} ({pca_nb_components}) - UMAP (n_neighbors={nb_neighbor}, min_dist={min_dist}, n_components={nb_components}, metric={metric})',
                width=1500,
                height=1000
            )

        fig.show()
    else:
        print("No features were extracted!")


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------


if __name__=='__main__':  # Fixed: Added double underscores
    
    # Create an instance of the pipeline (not just assigning the class)
    few_shot = DinoSim_pipeline(model,
                                model.patch_size,
                                device,
                                get_img_processing_f(resize_size),
                                feat_dim, 
                                dino_image_size=resize_size
                                )

    k = 10
    d = 0.5
    
    good_idx = [5, 7, 8, 10, 13, 300, 310, 316, 337, 343, 611, 614, 622, 623, 631, 901, 903, 905, 907, 912, 1210, 1211, 1213, 1220, 1221, 1500, 1507, 1509, 1510, 1514]
    files = get_fnames() # returns list(zip(np.concatenate(files), np.concatenate(labels)))
    good_files = [files[idx][0] for idx in good_idx] # len = 30
    datasets = [good_files[i:i+5] for i in range(0, len(good_idx), 5)] # list of lists of files
    good_labels = [files[idx][1] for idx in good_idx]
    _labels = [good_labels[i:i+5] for i in range(0, len(good_idx), 5)] # list of lists of labels
    
    latent_list, label_list = [], []

    for dataset, batch_label in tqdm(zip(datasets, _labels), desc='Iterating through neurotransmitters'):

        # Load images and prepare data
        images = np.array([load_image(file)[0] for file in dataset]).transpose(0,2,3,1)  # Convert to numpy array
        coordinates = [(0, load_image(file)[1], load_image(file)[2]) for file in dataset]

        # Pre-compute embeddings
        few_shot.pre_compute_embeddings(
            images,  # Pass numpy array of images
            overlap=(0.5, 0.5),
            padding=(0, 0),
            crop_shape=(512, 512, 1),
            verbose=True,
            batch_size=5
        )
        
        # Set reference vectors
        few_shot.set_reference_vector(coordinates, filter=None)
        
        # Get closest elements - using the correct method name
        
        close_embedding = few_shot.get_k_closest_elements(
            k=k
        )
        k_labels = [batch_label[0] for _ in range(k)]
        '''
        close_embedding, _, _ = few_shot.get_d_closest_elements(
            d=0.05,
            verbose=True
        )'''
        
        # Convert to numpy for storing
        close_embedding_np = close_embedding.cpu().numpy() if isinstance(close_embedding, torch.Tensor) else close_embedding
        
        latent_list.append(close_embedding_np)
        label_list.append(k_labels)
        
        # Clean up to free memory
        few_shot.delete_precomputed_embeddings()
        few_shot.delete_references()
    
    # Stack all embeddings and labels
    latents = np.vstack(latent_list)  # Changed from stack to vstack for proper concatenation
    labels = np.hstack(label_list)    # Changed from stack to hstack for proper concatenation

    print('Preparing display')

    # Display features
    diplay_features(
        latents,
        labels,
        include_pca=False,
        pca_nb_components=50,
        clustering=False,
        nb_clusters=6,
        nb_neighbor=5,
        min_dist=0.05,
        nb_components=2,
        metric='cosine'
    )
