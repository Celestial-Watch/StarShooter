import tkinter as tk
from tkinter import PhotoImage
from glob import glob
import csv
import os

# Initialize Tkinter root widget
root = tk.Tk()
root.title("30x30 Image with Buttons")

# Set the geometry (width x height + Xoffset + Yoffset)
root.geometry("1600x900+0+0")

# Load the 30x30 image
images_path = glob("../../data/images/centered_on_asteroid/*")
# images_path = sorted(images_path, reverse=True)

image_idx = 0
image = PhotoImage(file=images_path[image_idx])
# Upscale the image
upscaled_image = image.zoom(8, 8)
upscaled_image_label = tk.Label(root, image=upscaled_image)
upscaled_image_label.pack()

# Create a frame to hold the buttons
button_frame = tk.Frame(root)
button_frame.pack()

index_to_label = {
    0: "Burn",
    1: "Cosmic Ray",
    2: "Noise",
    3: "Scar",
    4: "Diffraction Spike",
    5: "Astronomical Object",
    6: "Streak",
    7: "Bright Comet",
}

sample_images = {
    0: "../../data/images/centered_on_asteroid/b0532057-2.png",
    1: "",
    2: "../../data/images/centered_on_asteroid/b0222544-1.png",
    3: "../../data/images/centered_on_asteroid/b0512238-2.png",
    4: "",
    5: "../../data/images/centered_on_asteroid/b0531793-1.png",
    6: "",
    7: "../../data/images/centered_on_asteroid/b0546729-2.png",
}

# Define the CSV file path
csv_file = "/Users/robinjehn/Documents/repos/StarShooter/src/annotation/annotations.csv"

# Define the column names
fieldnames = ["file_name", "label"]

if not os.path.exists(csv_file):
    # Open the CSV file in write mode
    with open(csv_file, "w", newline="") as file:
        # Create a CSV writer object
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write the header row
        writer.writeheader()

# Create 8 buttons and add them to the frame
photos = []
for idx in range(0, 8):
    def get_func(idx):
        def save_annotation(index=idx):
            global image_idx
            annotation = index_to_label[index]
            image_file_path = images_path[image_idx]

            with open(csv_file, "a", newline="") as file:
                # Create a CSV writer object
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writerow(
                    {"file_name": os.path.basename(image_file_path), "label": idx}
                )
            print(f"Annotation saved: {annotation} for image {image_file_path}")

            # Update the image
            image_idx += 1
            new_image_file_path = images_path[image_idx]
            new_image = PhotoImage(file=new_image_file_path)
            upscaled_new_image = new_image.zoom(8, 8)
            upscaled_image_label.configure(image=upscaled_new_image)
            upscaled_image_label.image = upscaled_new_image
        return save_annotation

    photos.append(PhotoImage(file=sample_images[idx]).zoom(4, 4))

    button = tk.Button(
        button_frame,
        text=index_to_label[idx],
        command=get_func(idx),
        image=photos[-1],
        compound=tk.TOP,
    )
    button.pack(side=tk.LEFT, padx=5)

# Run the application
root.mainloop()
