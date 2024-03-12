from utils import get_dataset, get_dataframe, get_engineered_features, CustomDataset
import torch
import model_def
import matplotlib.pyplot as plt
import os

model_path = "model/end-to-end-max_movement_vector_distance_20240303_222906/model_20240303_222906_1"
engineered_feature = "max_movement_vector_distance"

# Load model
images_per_sequence = 4
feature_vector_size = 10
image_shape = (30, 30)

if engineered_feature == "other_metadata":
    required_metadata = [
        "BackgroundMean",
        "BackgroundSigma",
        "MagnitudeZeroPoint",
        "AverageResidual",
        "RmsResidual",
        "FitOrder",
        "pos_Flux",
        "pos_Magnitude",
    ]
elif engineered_feature == "no_metadata":
    required_metadata = []
else:
    required_metadata = ["pos_RightAscension", "pos_Declination"]

path_to_data = os.path.abspath("./../../data/")
real_movers_file = f"{path_to_data}/csv/movers_cond_12_image_meta_data.csv"
bogus_movers_file = f"{path_to_data}/csv/movers_cond_2_image_meta_data.csv"
images_folder = f"{path_to_data}/images/centered_on_asteroid/"
movers_agg = get_dataframe(real_movers_file, bogus_movers_file, required_metadata)
data_set, mover_ids = get_dataset(movers_agg, images_folder)

# Get engineered features
movers_agg = movers_agg.filter(lambda x: any(x["mover_id"].isin(mover_ids))).groupby(
    "mover_id"
)
metadata = get_position_tensor(movers_agg)
extra_features = get_engineered_features(metadata, engineered_feature)
data_set = CustomDataset(data_set.tensors[0], extra_features, data_set.tensors[1])

model = torch.load(model_path)
model.eval()
print(f"Model: {model}")

fp = 0
fn = 0
plot = False
for idx, data in enumerate(data_set):
    x_tup, label = data
    x, feature_vector = x_tup
    x = x[None, :, :, :]
    feature_vector = feature_vector[None, :]
    pred = model((x, feature_vector))
    pred_label = int(pred.item() > 0.5)
    if pred_label != label.item():
        if pred_label == 1:
            fp += 1
        else:
            fn += 1

        mover_id = mover_ids[idx]
        text = f"Prediction for {mover_id}: {pred_label} ({round(pred.item(), 2)}), Label: {label.item()}"
        print(text)
        print(f"Feature vector: {feature_vector}\n")
        if plot:
            fig, axs = plt.subplots(1, 4)
            images = torch.split(x, 30, dim=2)
            fig.suptitle(text)
            for i, image in enumerate(images):
                axs[i].imshow(image.squeeze().numpy())
            plt.show()

precision = 1 - (fp / (fp + fn))
recall = 1 - (fn / (fp + fn))
f1_score = 2 * precision * recall / (precision + recall)
accuracy = (len(data_set) - (fp + fn)) / len(data_set)
print(f"The model was trained with the {engineered_feature} feature.")
print(f"{fp} false positives, {fn} false negatives out of {len(data_set)} samples.")
print(f"False positive rate: {fp / len(data_set)}")
print(f"False negative rate: {fn / len(data_set)}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 score: {f1_score}")
