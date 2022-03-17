import os
import glob

# Get Files Per Flag
files_per_flag = [len(glob.glob(filepath + "/*")) for filepath in glob.glob("data/images_country_flags_modified/*")]
print(files_per_flag)

# Get min Files
min_files = min(files_per_flag)

files_seen = 0
# Get All File Names for each continent
for flag_path in glob.glob("data/images_country_flags_modified/*"):
    # Get Images in Each Continent
    images_per_flag = glob.glob(flag_path + "/*")
    # Delete Images at End
    for image_loc in images_per_flag[min_files:]:
        os.remove(image_loc)
        files_seen += 1

        if files_seen % 1000 == 0:
            print(f"Removed {files_seen} files.")
    