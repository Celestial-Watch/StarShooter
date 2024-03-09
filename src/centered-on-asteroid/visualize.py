from utils import get_dataset, get_dataframe
import torch
import model_def
import matplotlib.pyplot as plt
from config import CSV_FOLDER, COA_IMAGE_FOLDER

model_path = "/Users/robinjehn/Documents/repos/StarShooter/src/centered-on-asteroid/model/end-to-end_20240216_161358/model_20240216_161358_4"

# Load model
images_per_sequence = 4
feature_vector_size = 10
image_shape = (30, 30)

model = model_def.CFN(images_per_sequence, feature_vector_size, image_shape)
model.load_state_dict(torch.load(model_path))
model.eval()

df = get_dataframe(CSV_FOLDER)
data_set, mover_ids = get_dataset(df, COA_IMAGE_FOLDER)

wrong_predictions = 0
for idx, data in enumerate(data_set):
    x, label = data
    x = x[None, :, :, :]
    pred = model(x)
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
