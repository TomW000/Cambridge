import torch
import numpy as np
from FragmentIdentifier import FragmentIdentifier
from processSkeletonData import generateTrainingData, getNeuronClassIndicesAndSkeletons, convertAttentionMaskToTrainingForm
import matplotlib.pyplot as plt
from tps import ThinPlateSpline
# srun --gres=gpu:1 --mem=128G --partition=ml --time=24:00:00 --pty tcsh
# srun --gres=gpu:1 --mem=128G --partition=ml --nodelist=fmg104 --time=10:00:00 --pty tcsh
torch.random.manual_seed(0)
np.random.seed(0)

# Load neuron class indices and skeletons
neuron_skeletons = getNeuronClassIndicesAndSkeletons()

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

# Model parameters
d_model = 60
num_heads = 6
num_layers = 6
d_ff = 256

# Define training parameters
n_iterations = 50000
n_epochs_to_display = 10
update_training_data_every_n_iterations = 1
lr = .001 # 3e-4
training_type = "minibatch_" # "minibatch_"
n_examples_per_mini_batch = 5

# ----------------------------------------------
# Load neuron fragments data
# ----------------------------------------------

# Define size of prediction output
neuron_skids = list(neuron_skeletons.keys())
n_neuron_types = len(neuron_skids)
prediction_output_size = [n_neuron_types,1]

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
# Run model
# ----------------------------------------------

# Define model
num_spatial_dimensions = 3
model = FragmentIdentifier(num_spatial_dimensions, d_model, num_heads, num_layers, n_neuron_types, d_ff=d_ff)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Move model to GPU
model = model.to(device)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Train model
model.losses = []
# lossFunc = torch.nn.CrossEntropyLoss()
for i in range(n_iterations):
    # Update training data
    if i % update_training_data_every_n_iterations == 0:
        # Get random neuron class indices
        random_indices = np.random.choice(n_neuron_types, max_examples_per_training_run, replace=True)
        random_skids = [neuron_skids[idx] for idx in random_indices]
        # Generate training data
        training_data, target_data, attention_mask = generateTrainingData(neuron_skeletons, random_skids, random_indices, n_fragments_per_neuron, fragment_max_size, n_neuron_types, min_fragment_radius, fragment_radius_var, gridifying_factor)
        # Convert attention mask to training form
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
    # Zero gradients
    optimizer.zero_grad()
    # Training run
    if training_type == "minibatch_":
        loss = torch.zeros(1).to(device)
        for t in range(0, training_data_transformed.shape[0], n_examples_per_mini_batch):
            # Forward pass
            output = model.forward(training_data_transformed[t:t+n_examples_per_mini_batch], mask=attention_mask[t:t+n_examples_per_mini_batch])
            # Compute loss
            loss += torch.nn.functional.binary_cross_entropy(output, target_data[t:t+n_examples_per_mini_batch])
        # Average loss
        loss /= (training_data_transformed.shape[0] / n_examples_per_mini_batch)
    else:
        # Forward pass
        output = model.forward(training_data_transformed, mask=attention_mask)
        # Compute loss
        loss = torch.nn.functional.binary_cross_entropy(output, target_data)
    # Backward pass
    loss.backward()
    # Optimizer step
    optimizer.step()
    # Print loss
    model.losses.append(loss.item())
    # Print loss
    if i % n_epochs_to_display == 0:
        print(f"Iteration {i}, loss: {np.mean(model.losses[-n_epochs_to_display:])}")

# Save model
torch.save(model, f"./trainedModels/{training_type}{n_iterations}_model_dModel{d_model}_dFF{d_ff}_heads{num_heads}_layers{num_layers}_{non_rigid_transfomation_scale}transScale_{n_iterations}iterations.pth")

# ----------------------------------------------
# Plot model performance
# ----------------------------------------------

# Plot model loss
losses = np.array(model.losses)
meanedLosses = []
Xs = []
for i in range(0, len(losses), n_epochs_to_display):
    meanedLosses.append(np.mean(losses[i:i+n_epochs_to_display]))
    Xs.append(i)

plt.close()
plt.plot(Xs, meanedLosses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig(f"model_loss_{training_type}{n_iterations}iterations_{d_model}dModel_{d_ff}dFF_{num_heads}heads_{num_layers}layers_{non_rigid_transfomation_scale}transScale.png", dpi=300)

# Test model performance
training_data, target_data, attention_mask = generateTrainingData(neuron_skeletons, random_indices, n_fragments_per_neuron, fragment_max_size, n_neuron_types, min_fragment_radius, fragment_radius_var, gridifying_factor, shuffle=False)
training_data = training_data.to(device)
target_data = target_data.to(device)

# Forward pass
output = model.forward(training_data)
output = output.cpu().detach().numpy()
target_data = target_data.cpu().numpy()

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
    
