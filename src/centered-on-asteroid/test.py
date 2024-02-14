import model_def
import torch
import pandas as pd
import torchvision
import os
from PIL import Image

image_shape = (30, 30)
images_per_sequence = 4
feature_vector_size = 10

path_to_data = os.path.abspath("./../../data/") + "/"

# Read csv
real_movers = pd.read_csv(path_to_data + "mover_images_lookup.csv")
bogus_movers = pd.read_csv(path_to_data + "rejected_mover_images_lookup.csv")

# Add labels
real_movers["label"] = 1
bogus_movers["label"] = 0

# Group by mover
movers = pd.concat([real_movers, bogus_movers])
movers_agg = movers.groupby("Mover_id")

# Generate input, output pairs
x_tensors = []
y_hat_tensors = []
for group_name, group_data in movers_agg:
    image_tensors = []
    for index, row in group_data.iterrows():
        image_path = path_to_data + "30x30_images/" + row["Name"]

        # Read image as PIL Image
        image = Image.open(image_path).convert("L")

        # Convert PIL Image to torch.Tensor
        transform = torchvision.transforms.ToTensor()
        image_tensor = transform(image)

        # Reshape image tensor to match the expected input shape
        image_tensor = image_tensor.view(1, 1, *image_shape)
        image_tensors.append(image_tensor)
    x_tensor = torch.cat(image_tensors, dim=2)
    x_tensors.append(x_tensor)
    y_hat_tensors.append(torch.tensor([[group_data["label"].iloc[0]]]))

x = torch.concat(x_tensors)
y_hat = torch.concat(y_hat_tensors)

model = model_def.MNN(images_per_sequence, feature_vector_size, image_shape)
model.load_state_dict(torch.load("model/model.pt"))
model.eval()

y = model(x)
print(torch.abs(y - y_hat))
