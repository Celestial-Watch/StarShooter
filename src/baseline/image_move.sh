source_dir="../processing/data/alistair/movers_2021-2024"
target_dir="../processing/data/alistair/30x30_images"

# Create the target directory if it doesn't exist
mkdir -p "$target_dir"

# Move all the images from the source directory to the target directory
for image in "$source_dir"/*; do
    filename=$(basename "$image")

    new_filename="${filename##*\\}"

    echo "Moving $filename to $target_dir/$new_filename"
    # Uncomment the next line to actually move the files
    mv "$image" "$target_dir/$new_filename"
done
echo "Successfully moved all images to $target_dir"