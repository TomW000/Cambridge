import torch
import numpy as np
from tqdm import tqdm
from FragmentIdentifier import FragmentIdentifier
from processSkeletonData import generateTrainingData, getNeuronClassIndicesAndSkeletons, convertAttentionMaskToTrainingForm
import matplotlib.pyplot as plt
from tps import ThinPlateSpline
# srun --gres=gpu:1 --mem=128G --partition=ml --time=24:00:00 --pty tcsh
# srun --gres=gpu:1 --mem=128G --partition=ml --nodelist=fmg104 --time=10:00:00 --pty tcsh

# Load neuron class indices and skeletons
neuron_skeletons = getNeuronClassIndicesAndSkeletons()
n_neuron_types = len(neuron_skeletons)
neuron_skids = list(neuron_skeletons.keys())
n_neuron_types = len(neuron_skids)

# ----------------------------------------------
# Define key parameters
# ----------------------------------------------

# Define number of fragments per neuron
n_transform_points = 4
n_fragments_per_neuron = 1
fragment_max_size = 500
max_examples_per_training_run = 100
non_rigid_transfomation_scale = 0.05
gridifying_factor = 0.1
min_fragment_radius = .1
fragment_radius_var = .1
training_type = "minibatch_" # "minibatch_"

# Model parameters
d_model = 60
num_heads = 6
num_layers = 6
d_ff = 256

# Define model
n_iterations = 50000
num_spatial_dimensions = 3
model = FragmentIdentifier(num_spatial_dimensions, d_model, num_heads, num_layers, n_neuron_types, d_ff=d_ff)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load from saved
model_fname = f"./trainedModels/{training_type}{n_iterations}_model_dModel{d_model}_dFF{d_ff}_heads{num_heads}_layers{num_layers}_{non_rigid_transfomation_scale}transScale_{n_iterations}iterations.pth"
model = torch.load(model_fname, map_location=torch.device(device))

# ----------------------------------------------
# Normalise neuron skeletons
# ----------------------------------------------

# Hard-code minumum and maximum values (for consistency)
min_vals = np.array([2500, 2000, 5000])
max_vals = np.array([100000, 100000, 250000])
mean_vals = np.array([np.mean([min_vals[i], max_vals[i]]) for i in range(3)])

# Normalise skeletons
for n_type in neuron_skeletons:
    for i, skel in enumerate(neuron_skeletons[n_type]):
        # neuron_skeletons[n_type][i] = (skel - min_vals) / (max_vals - min_vals)
        neuron_skeletons[n_type][i] = (skel - mean_vals) / (max_vals - min_vals)

# Define function to deform skeletons
def generateRandomNonRigidTransform(n_points, transformation_scale):
    tps = ThinPlateSpline(alpha=1.0)
    control_points = np.random.rand(n_points, 3)
    target_points = control_points + np.random.uniform(-transformation_scale, transformation_scale, 3)
    tps.fit(control_points, target_points)
    return tps

# Get skeleton ranges
skeleton_ranges = {}
for n_type in neuron_skeletons:
    skeleton_ranges[n_type] = []
    for skel in neuron_skeletons[n_type]:
        skeleton_ranges[n_type].append([np.min(skel, axis=0), np.max(skel, axis=0)])

# Get min and max values for each dimension
min_vals = np.inf
max_vals = -np.inf
for n_type in neuron_skeletons:
    for skel in neuron_skeletons[n_type]:
        min_vals = np.minimum(min_vals, np.min(skel, axis=0))
        max_vals = np.maximum(max_vals, np.max(skel, axis=0))


# ----------------------------------------------
# Define functions for running model
# ----------------------------------------------

def applyRandomNonRigidTransformation(training_data, target_data, attention_mask, n_transform_points, non_rigid_transfomation_scale):
    # Apply random non-rigid transformation
    tps = generateRandomNonRigidTransform(n_transform_points, non_rigid_transfomation_scale)
    training_data_transformed = np.zeros_like(training_data)
    training_data_np = training_data.cpu().numpy()
    for n in range(training_data.shape[0]):
        training_data_transformed[n] = tps.transform(training_data_np[n])
    training_data_transformed = torch.tensor(training_data_transformed, dtype=torch.float32)
    return training_data_transformed, target_data, attention_mask

def getTrainingData(neuron_skeletons, skids, indices, n_neuron_types, n_fragments_per_neuron, fragment_max_size, max_examples_per_training_run, min_fragment_radius, fragment_radius_var, gridifying_factor):
    # Generate training data
    training_data, target_data, attention_mask = generateTrainingData(neuron_skeletons, skids, indices, n_fragments_per_neuron, fragment_max_size, n_neuron_types, min_fragment_radius, fragment_radius_var, gridifying_factor)
    attention_mask = convertAttentionMaskToTrainingForm(attention_mask, max_examples_per_training_run, num_heads, fragment_max_size, device)
    training_data_transformed, target_data, attention_mask = applyRandomNonRigidTransformation(training_data, target_data, attention_mask, n_transform_points, non_rigid_transfomation_scale)
    # Move data to GPU
    training_data_transformed = training_data_transformed.to(device)
    target_data = target_data.to(device)
    return training_data, target_data, attention_mask

random_indices = np.random.choice(n_neuron_types, max_examples_per_training_run, replace=True)
random_skids = [neuron_skids[idx] for idx in random_indices]
training_data, target_data, attention_mask = getTrainingData(neuron_skeletons, random_skids, random_indices, n_neuron_types, n_fragments_per_neuron, fragment_max_size, max_examples_per_training_run, min_fragment_radius, fragment_radius_var, gridifying_factor)

# ----------------------------------------------
# Look at example fragments
# ----------------------------------------------

# Generate training data
random_indices = np.random.choice(n_neuron_types, max_examples_per_training_run, replace=True)
random_skids = [neuron_skids[idx] for idx in random_indices]
training_data, target_data, attention_mask = generateTrainingData(neuron_skeletons, random_skids, random_indices, n_fragments_per_neuron, fragment_max_size, n_neuron_types, min_fragment_radius, fragment_radius_var, gridifying_factor)
attention_mask = convertAttentionMaskToTrainingForm(attention_mask, max_examples_per_training_run, num_heads, fragment_max_size, device)

# Apply random non-rigid transformation
tps = generateRandomNonRigidTransform(n_transform_points, non_rigid_transfomation_scale)
training_data_transformed = np.zeros_like(training_data)
training_data_np = training_data.cpu().numpy()
for n in range(training_data.shape[0]):
    training_data_transformed[n] = tps.transform(training_data_np[n])

training_data_transformed = torch.tensor(training_data_transformed, dtype=torch.float32)

# Move data to GPU
training_data_transformed = training_data_transformed.to(device)
target_data = target_data.to(device)

# Run model
threshold = .05
accuracies = np.zeros(max_examples_per_training_run)
threshold_accuracies = np.zeros(max_examples_per_training_run)
true_label_probabilities = np.zeros(max_examples_per_training_run)
predicted_label_probabilities = np.zeros(max_examples_per_training_run)
true_label_rankings = np.zeros(max_examples_per_training_run)
for i in range(max_examples_per_training_run):
    output = model.forward(training_data_transformed[[i]], mask=attention_mask[[i]])
    output = output.cpu().detach().numpy()
    true_class_id = np.argmax(target_data[i].cpu().numpy())
    predicted_class_id = np.argmax(output, axis=1)
    true_skeleton_id = neuron_skids[true_class_id]
    predicted_skeleton_id = neuron_skids[predicted_class_id[0]]
    accuracies[i] = 1 if true_skeleton_id == predicted_skeleton_id else 0
    threshold_accuracies[i] = 1 if output[0,true_class_id] > threshold else 0
    true_label_probabilities[i] = output[0,true_class_id]
    predicted_label_probabilities[i] = output[0,predicted_class_id[0]]
    true_label_rankings[i] = np.where(np.argsort(output[0,:])[::-1]==true_class_id)[0][0]

# Get fragment colours
colours = ["blue" if a==1 else "red" for a in accuracies]

# Plot true label rankings
plt.close()
plt.scatter(np.arange(len(true_label_rankings)), true_label_rankings, c=colours)
plt.xlabel("Fragment index")
plt.ylabel("True label ranking")
plt.ylim([n_neuron_types,-100])
plt.box("off")
plt.savefig("./figures/modelTesting_trueLabelRankings.png", dpi=300)

# Plot true and predicted label probabilities
plt.close()
plt.scatter(true_label_probabilities, predicted_label_probabilities, c=colours)
plt.xlabel("True label probability")
plt.ylabel("Predicted label probability")
plt.savefig("./figures/modelTesting_trueAndPredictedLabelProbabilities.png", dpi=300)

# Print accuracies
print(f"Accuracy: {np.mean(accuracies*100)}")
print(f"{threshold} threshold accuracy: {np.mean(threshold_accuracies*100)}")

# ----------------------------------------------
# Look at example fragments
# ----------------------------------------------
import napari as nap

# Get training data
n_examples = 100
training_data, target_data, attention_mask = getTrainingData(neuron_skeletons, n_neuron_types, n_fragments_per_neuron, fragment_max_size, n_examples, min_fragment_radius, fragment_radius_var, gridifying_factor)

# Find example of specific cell type
cell_type_index = 2
fragment_indices = np.where(target_data.cpu().numpy()[:,cell_type_index]==1)[0][0]

# Forward pass
output = model.forward(training_data)
output = output.cpu().detach().numpy()

n_neurons_to_plot = 20
v = nap.Viewer()
for i in range(n_neurons_to_plot):
    cur_skeleton = training_data[i].cpu().numpy()
    valid_values = cur_skeleton[:,0] != 0
    cur_skeleton = cur_skeleton[valid_values]
    v.add_points(cur_skeleton*500, size=2, name=f"Fragment {i}", face_color=np.random.rand(3), blending="translucent")

# ----------------------------------------------
# Test model ability to select only fragments from single cell type
# ----------------------------------------------

# Get training data for just a few cell types
class_ids = [2,3]
subsampled_neuron_skeletons = [neuron_skeletons[c] for c in class_ids]
n_fragments_per_neuron = 100
training_data, target_data, attention_mask = getTrainingData(neuron_skeletons, len(class_ids), n_fragments_per_neuron, fragment_max_size, len(class_ids), min_fragment_radius, fragment_radius_var, gridifying_factor)

# Apply random non-rigid transformation
training_data_transformed, target_data, attention_mask = applyRandomNonRigidTransformation(training_data, target_data, attention_mask, n_transform_points, non_rigid_transfomation_scale)

# Forward pass
output = model.forward(training_data)
output = output.cpu().detach().numpy()

r = np.argmax(output[1,:])
print(class_skids[r])

# Get model performance on each fragment
predicted_label_probabilities = np.zeros((output.shape[0], 1))
for i in range(output.shape[0]):
    target = target_data[i]
    target_label = class_ids[np.where(target==1)[0][0]]
    predicted_label_probability = output[i,target_label]
    predicted_label_probabilities[i] = predicted_label_probability

plt.hist(predicted_label_probabilities, bins=20)
plt.show()


# Test accuracy with different transformation scales
accuracies = []
n_examples = 500
non_rigid_transfomation_scales = [0, 0.05, 0.1, 0.15, 0.2, 0.25]#, 0.30]
for t_scale in tqdm(non_rigid_transfomation_scales):
    # Get training data
    training_data, target_data, attention_mask = getTrainingData(neuron_skeletons, n_neuron_types, n_fragments_per_neuron, fragment_max_size, n_examples, min_fragment_radius, fragment_radius_var, gridifying_factor)
    # Apply random non-rigid transformation
    training_data_transformed, target_data, attention_mask = applyRandomNonRigidTransformation(training_data, target_data, attention_mask, n_transform_points, non_rigid_transfomation_scale=t_scale)
    # Forward pass
    output = model.forward(training_data_transformed)
    output = output.cpu().detach().numpy()
    target_data = target_data.cpu().numpy()
    # Get model performance
    cur_accuracies = []
    for i in range(output.shape[0]):
        cur_output, cur_target = output[i], target_data[i]
        cur_max_index = np.argmax(cur_output, axis=0)
        cur_target_label = np.where(cur_target==1)[0][0]
        accuracy = 1 if cur_max_index == cur_target_label else 0
        cur_accuracies.append(accuracy)
    # Save accuracy
    accuracies.append(np.mean(cur_accuracies))

plt.bar(np.arange(len(accuracies)), accuracies)
plt.xticks(np.arange(len(accuracies)), non_rigid_transfomation_scales)
plt.show()

print(f"Accuracy: {np.mean(cur_accuracies)}")

asjkhd





# ----------------------------------------------
# Plot model performance
# ----------------------------------------------

# Test model performance
training_data, target_data, attention_mask = generateTrainingData(neuron_skeletons, random_indices, n_fragments_per_neuron, fragment_max_size, n_neuron_types, min_fragment_radius, fragment_radius_var, gridifying_factor, shuffle=False)
training_data = training_data.to(device)
target_data = target_data.to(device)



onProbabilities = []
offProbabilities = []
accuracies = []
for i in range(output.shape[0]):
    cur_output = output[i]
    cur_target = target_data[i]
    cur_max_index = np.argmax(cur_output, axis=0)
    cur_target_label = np.where(cur_target==1)[0][0]
    cur_nontarget_label = np.where(cur_target==0)[0][0]
    accuracy = 1 if cur_max_index == cur_target_label else 0
    onProbability = np.mean(cur_output[cur_target_label])
    offProbability = np.mean(cur_output[cur_nontarget_label])
    onProbabilities.append(onProbability)
    offProbabilities.append(offProbability)
    accuracies.append(accuracy)

plt.close()
plt.hist(np.subtract(onProbabilities,offProbabilities), bins=20, alpha=.5, label="On")
# plt.hist(offProbabilities, bins=20, alpha=.5, label="Off")
plt.legend()
plt.savefig("model_performance_histogram.png", dpi=300)

with torch.no_grad():
    # Define transformation scales
    transformation_scales = [0, 0.05, 0.1, 0.5, 1, 2]
    fig,ax = plt.subplots(2,len(transformation_scales),figsize=(10,5),sharex=True,sharey=True)
    # Iterate over transformation scales
    for i,transformation in enumerate(transformation_scales):
        # Generate training data
        training_data, target_data, attention_mask = generateTrainingData(neuron_skeletons, random_indices, n_fragments_per_neuron, fragment_max_size, n_neuron_types, min_fragment_radius, fragment_radius_var, gridifying_factor, shuffle=False)
        # Apply random non-rigid transformation
        tps = generateRandomNonRigidTransform(n_transform_points, transformation_scale=transformation)
        training_data_transformed = np.zeros_like(training_data)
        training_data_np = training_data.cpu().numpy()
        for n in range(training_data.shape[0]):
            training_data_transformed[n] = tps.transform(training_data_np[n])
        training_data_transformed = torch.tensor(training_data_transformed, dtype=torch.float32)
        # Move data to GPU
        training_data_transformed = training_data_transformed.to(device)
        target_data = target_data.to(device)
        # Forward pass
        output = model.forward(training_data_transformed)
        # Plot results
        loss = torch.nn.functional.binary_cross_entropy(output, target_data)
        ax[0,i].imshow(target_data.cpu().numpy(), vmin=0, vmax=1)
        ax[0,i].set_title(f"T = {transformation:.2f}\n(Loss = {loss.item():.4f})")
        ax[1,i].imshow(output.cpu().numpy(), vmin=0, vmax=1, aspect="auto")
        _ = [ax[j,i].axis("off") for j in range(2)]
    plt.tight_layout()
    plt.savefig("model_performance.png", dpi=300)
    

# src = training_data_transformed
# # src shape: (N_examples, N_points, 3_dimensions)
# src = src.permute(1, 0, 2)  # Change to (N_points, N_examples, 3_dimensions)
# src = model.embedding(src)
# src = model.pos_encoder(src)
# src = model.dropout(src)
# x = src

# attn = model.layers[0].self_attn


# batch_size = x.shape[1]
# Q = attn.W_q(x).view(batch_size, -1, attn.num_heads, attn.d_k).transpose(1, 2)
# K = attn.W_k(x).view(batch_size, -1, attn.num_heads, attn.d_k).transpose(1, 2)
# V = attn.W_v(x).view(batch_size, -1, attn.num_heads, attn.d_k).transpose(1, 2)
# output = attn.scaled_dot_product_attention(Q, K, V, attention_mask)
# output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)



# attn_output = layer.self_attn(x, x, x, attention_mask)
# attn_output = torch.swapaxes(attn_output, 0, 1)
# x = layer.norm1(x + layer.dropout(attn_output))
# ff_output = la.feed_forward(x)
# x = self.norm2(x + self.dropout(ff_output))

# attn = model.layers[0]







# if attention_mask is not None:
#     attn_scores = attn_scores.masked_fill(attention_mask == 0, -torch.inf)
# attn_probs = torch.softmax(attn_scores, dim=-1)
# output = torch.matmul(attn_probs, V)

# for layer in model.layers:
#     src = layer(src, attention_mask)