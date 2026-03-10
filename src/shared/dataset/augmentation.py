import os
import shutil
import tarfile
import xml.etree.ElementTree as ET
import re
from astropy.io import fits
import numpy as np
from shared.dataset.dl_util import extract_tar

dest_path=f"/raid/persistent_scratch/schlehan/jobs/fits_test"
ann_tar_path = f"{dest_path}/annotation.tar" # Path where annotation (.xml) files are stored
img_tar_path = f"{dest_path}/FITSImages.tar"  # Path where image (.fits) files are stored

ann_path = f"{dest_path}/annotation/" # Path where annotation (.xml) files are stored
img_path = f"{dest_path}/FITSImages/"  # Path where image (.fits) files are stored

copy_ann = f"{dest_path}/AugementedAnnotations/"  # Destination path to copy valid annotation files
copy_img = f"{dest_path}/AugmentedFITSImages/"  # Destination path to copy valid image files

def compress_to_tar(output_tar_path, directory):
    with tarfile.open(output_tar_path, "w") as tar:
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    tar.add(file_path, arcname=os.path.relpath(file_path, start=os.path.dirname(directory)))
                except PermissionError:
                    print(f"Skipping {file_path} due to permission error.")
    print(f"Compressed files to {output_tar_path}")

extract_tar(ann_tar_path, ann_path)
extract_tar(img_tar_path, img_path)
    
# Regex pattern to match filenames starting with 'FIRSTJ' followed by specific format
pattern = re.compile(r"^FIRSTJ[0-9]{6}\.[0-9]\+[0-9]{6}\.xml$")

dropped_imgs = 0

for xml_file in os.listdir(ann_path):
    if pattern.match(xml_file):
        xml_path = os.path.join(ann_path, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        objects = root.findall("object")

        # Skip if more than one object is present
        if len(objects) != 1:
            dropped_imgs += 1
            continue

        img_name = root.find("filename").text

        # Ensure the image name ends with .fits
        img_name = img_name.rsplit(".", 1)[0] + ".fits"

        # Paths to source image and annotation
        img_src = os.path.join(f"{img_path}/FITSImages", img_name)
        xml_dest = os.path.join(copy_ann, xml_file)
        img_dest = os.path.join(copy_img, img_name)
        # Copy annotation file
        shutil.copy2(xml_path, xml_dest)

        if os.path.exists(img_src):
                    shutil.copy2(img_src, img_dest)

                    # If the object name is '1_2', create a mirrored copy
                    object_name = objects[0].find("name").text
                    if object_name == "1_2":
                        # Load and mirror the resnet_fits image
                        with fits.open(img_src) as hdul:
                            mirrored_data = np.flip(hdul[0].data, axis=1)
                            hdul[0].data = mirrored_data

                            mirrored_img_name = img_name.replace(".fits", "M.fits")
                            mirrored_img_dest = os.path.join(copy_img, mirrored_img_name)
                            hdul.writeto(mirrored_img_dest, overwrite=True)

                        # Create a new mirrored XML file
                        mirrored_xml_name = xml_file.replace(".xml", "M.xml")
                        mirrored_xml_dest = os.path.join(copy_ann, mirrored_xml_name)

                        mirrored_tree = ET.parse(xml_path)
                        mirrored_root = mirrored_tree.getroot()
                        mirrored_root.find("filename").text = mirrored_img_name

                        mirrored_tree.write(mirrored_xml_dest)

                        # Create a 90-degree rotated version of the mirrored image
                        with fits.open(mirrored_img_dest) as hdul:
                            mirrored_rotated_data = np.rot90(hdul[0].data)
                            hdul[0].data = mirrored_rotated_data

                            mirrored_rotated_img_name = img_name.replace(".fits", "MR.fits")
                            mirrored_rotated_img_dest = os.path.join(copy_img, mirrored_rotated_img_name)
                            hdul.writeto(mirrored_rotated_img_dest, overwrite=True)

                        # Create a new XML file for the mirrored-rotated image
                        mirrored_rotated_xml_name = xml_file.replace(".xml", "MR.xml")
                        mirrored_rotated_xml_dest = os.path.join(copy_ann, mirrored_rotated_xml_name)

                        mirrored_rotated_tree = ET.parse(xml_path)
                        mirrored_rotated_root = mirrored_rotated_tree.getroot()
                        mirrored_rotated_root.find("filename").text = mirrored_rotated_img_name

                        mirrored_rotated_tree.write(mirrored_rotated_xml_dest)

                    # 90 grad rotation bei name != 1_1
                    if object_name != "1_1":
                        # Load and rotate the resnet_fits image
                        with fits.open(img_src) as hdul:
                            rotated_data = np.rot90(hdul[0].data)
                            hdul[0].data = rotated_data

                            rotated_img_name = img_name.replace(".fits", "R.fits")
                            rotated_img_dest = os.path.join(copy_img, rotated_img_name)
                            hdul.writeto(rotated_img_dest, overwrite=True)

                        # Create a new rotated XML file
                        rotated_xml_name = xml_file.replace(".xml", "R.xml")
                        rotated_xml_dest = os.path.join(copy_ann, rotated_xml_name)

                        rotated_tree = ET.parse(xml_path)
                        rotated_root = rotated_tree.getroot()
                        rotated_root.find("filename").text = rotated_img_name

                        rotated_tree.write(rotated_xml_dest)

        else:
            print(f"Warning: Image file not found - {img_src}")

print(f"Processing complete. Dropped images: {dropped_imgs}")

compress_to_tar(f"{dest_path}/AugementedAnnotations.tar", copy_ann)
compress_to_tar(f"{dest_path}/AugmentedFITSImages.tar", copy_img)