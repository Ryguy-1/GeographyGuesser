import os
import glob

# Get Files Per Continent
files_per_continent = [len(glob.glob(filepath + "/*")) for filepath in glob.glob("data/images_sorted_by_continent/*")]
print(files_per_continent)

# Get min Files
min_files = min(files_per_continent)

files_seen = 0
# Get All File Names for each continent
for continent_path in glob.glob("data/images_sorted_by_continent/*"):
    # Get Images in Each Continent
    images_per_continent = glob.glob(continent_path + "/*")
    # Delete Images at End
    for image_loc in images_per_continent[min_files:]:
        os.remove(image_loc)
        files_seen += 1

        if files_seen % 1000 == 0:
            print(f"Removed {files_seen} files.")
    