from utils import *
from model import *

if latent:
    all_features = torch.cat(latent, dim=0).numpy()

    pca = PCA(n_components=25)
    reduced_features = pca.fit_transform(all_features)
    

    n_neighbor=5
    min_dist=0.01
    n_components=2
    metric='cosine' #cosine
    random_state=10
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbor,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state
    )
    embedding = reducer.fit_transform(reduced_features)
    
    fig = px.scatter(
        x=embedding[:, 0],
        y=embedding[:, 1],
        color = label_list,
        title = f'UMAP with n_neighbors={n_neighbor}, min_dist={min_dist}, n_components={n_components}, metric={metric},random_state={random_state}',
        width=1500,
        height=1000
    )
    fig.show()
else:
    print("No features were extracted!")
    
    
knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=6)