from utils import (
    get_dataset,
    get_dataframe,
    get_position_tensor,
    get_engineered_features,
)
from train_with_meta_data import CustomDataset
import torch
import model_def
import matplotlib.pyplot as plt
import os

model_path = "/Users/robinjehn/Documents/repos/StarShooter/src/centered-on-asteroid/model/end-to-end_20240218_092303/model_20240218_092303_3"

# Load model
images_per_sequence = 4
feature_vector_size = 10
image_shape = (30, 30)
meta_data_size = 1

model = model_def.MCFN(
    images_per_sequence, feature_vector_size, image_shape, meta_data_size
)
model.load_state_dict(torch.load(model_path))
model.eval()

path_to_data = os.path.abspath("./../../data/")
real_movers_file = f"{path_to_data}/csv/movers_cond_12_image_meta_data.csv"
bogus_movers_file = f"{path_to_data}/csv/movers_cond_2_image_meta_data.csv"
images_folder = f"{path_to_data}/images/centered_on_asteroid/"
movers_agg = get_dataframe(real_movers_file, bogus_movers_file)
data_set, mover_ids = get_dataset(movers_agg, images_folder)
metadata = get_position_tensor(movers_agg)
# metadata = torch.fill(metadata, 0)
extra_features = get_engineered_features(metadata)
data_set = CustomDataset(data_set.tensors[0], extra_features, data_set.tensors[1])

wrong_predictions = 0
for idx, data in enumerate(data_set):
    x_tup, label = data
    x, feature_vector = x_tup
    x = x[None, :, :, :]
    feature_vector = feature_vector[None, :]
    pred = model((x, feature_vector))
    pred_label = round(pred.item())
    if pred_label != label.item():
        wrong_predictions += 1
        fig, axs = plt.subplots(1, 4)
        images = torch.split(x, 30, dim=2)
        mover_id = mover_ids[idx]
        text = f"Prediction for {mover_id}: {pred_label}, Label: {label.item()}"
        fig.suptitle(text)
        print(text)
        for i, image in enumerate(images):
            axs[i].imshow(image.squeeze().numpy())
        plt.show()

print(f"{wrong_predictions} wrong predictions out of {len(data_set)} samples.")
