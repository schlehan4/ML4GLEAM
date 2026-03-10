from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from astropy.io import fits
from shared.dataset.dl_util import *

img_path = "./data/FITSImages/FITSImages/"
coco_path = "./data/coco_annotation.json"
#
# # TODO: @schlehan, please put all the dataloader stuff in a "Dataloader" module -> /shared/Dataloader/*
# class CocoLoader(Dataset):
#     def __init__(self, img_folder,  annotation_file, transform=None):
#         self.img_folder = img_folder
#         self.transform = transform
#
#
#         with open(annotation_file, 'r') as f:
#             data = json.load(f)
#         self.image_id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}
#
#         self.annotations = []
#         for ann in data["annotations"]:
#             image_id = ann["id"] # TODO: here you have changed the key from "image_id" to "id" (gehrimor)
#             category_id = ann["category_id"]
#             self.annotations.append((image_id, category_id))
#
#         self.category_mapping = {cat["id"]: cat["name"] for cat in data["categories"]}
#
#     def __len__(self):
#         return len(self.annotations)
#
#     def __getitem__(self, index):
#         image_id, category_id = self.annotations[index]
#         image_filename = self.image_id_to_filename[image_id]
#         image_path = os.path.join(self.img_folder, image_filename)
#
#         image = Image.open(image_path).convert("RGB")
#         if self.transform:
#             image = self.transform(image)
#         label = category_id - 1
#         return image, label
    
class FitsSet(Dataset):
    def __init__(self, img_folder, xml_annotation_path, annotation_file, transform=None):
        self.img_folder = img_folder
        self.transform = transform
        self.xml_annotation_path = xml_annotation_path
        
        with open(annotation_file, 'r') as f:
            data = json.load(f)

        self.image_id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

        self.annotations = []
        for ann in data["images"]:
            image_id = ann["id"]
            # category_id = ann["category_id"]
            fits_filename = ann["file_name"]

            # print("Filename actual: ", fit_filename)
            
            # # 50 % der Kategorie 1-Bilder ausschließen
            # if category_id == 1 and random.random() < 0.635:
            #     continue
            xml_filename = fits_filename.replace(".fits", ".xml") # this is the filename of the original xml file with the annotations in raw data form
            self.annotations.append((image_id, fits_filename, xml_filename))

        self.category_mapping = {cat["id"]: cat["name"] for cat in data["categories"]}

    def __len__(self):
        return len(self.annotations)

    import os
    import xml.etree.ElementTree as ET

    def load_cat_from_xml(self,xml_path):
        """
        Lädt und parst eine XML-Datei im Pascal VOC Format und extrahiert die Kategorie.

        Args:
            xml_path (str): Pfad zur XML-Datei

        Returns:
            list: Liste der extrahierten Kategorien
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()

        categories = []
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            categories.append(class_name)

        return categories

    def get_id_from_cat(self, cat):
        """
        Gibt die ID einer Kategorie zurück.

        Args:
            cat (str): Kategoriename

        Returns:
            int: Die ID der Kategorie oder None, falls nicht gefunden.
        """
        category_mapping = {
            "1_1": 0,
            "1_2": 1,
            "3_3": 2,
            "1_3": 3,
            "2_2": 4,
            "2_3": 5
        }

        return category_mapping.get(cat, None)

    def __getitem__(self, index):
        image_id, fits_filename, xml_filename = self.annotations[index]
        # image_filename = self.image_id_to_filename[image_id]
        image_filename = fits_filename

        # load cat from xml directly
        xml_path = os.path.join(self.xml_annotation_path, xml_filename)
        category = None
        categories = self.load_cat_from_xml(xml_path)
        if len(categories) > 1:
            # print("Warning: Multiple categories in one image")
            # print("All Categories: ", categories)
            # print("xml Filename: ", xml_filename)
            return None
        else:
            category = categories[0]

        category_id = self.get_id_from_cat(category)

        image_path = os.path.join(self.img_folder, image_filename)
        # if random.random() < 0.01:
        #     print("image_path: ", image_path)
        #     print("Index: ", index)
        #     print("Category ID: ", category_id - 1)

        # TODO: here you need to load the raw xml annotations and read the category from there...

        # resnet_fits-Bild laden
        with fits.open(image_path) as hdul:
            image_data = hdul[0].data.astype(np.float32)

        # Ungültige Werte (NaN, Inf) behandeln
        image_data = np.nan_to_num(image_data, nan=0.0, posinf=1.0, neginf=0.0)

        # Normalisierung (0 bis 1)
        image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data) + 1e-8)
        
        # Auf 3 Kanäle erweitern (z. B. für ResNet)
        #empty_channel = np.zeros_like(image_data)
        #image_data = np.stack((image_data, empty_channel, empty_channel), axis=2)
        image_data = np.repeat(image_data[:, :, np.newaxis], 3, axis=2)

        # In PIL-Bild umwandeln
        image_data = Image.fromarray((image_data * 255).astype(np.uint8))

        # Transformation anwenden (falls angegeben)
        if self.transform:
            image_data = self.transform(image_data)

        # Label anpassen (0-basiert für PyTorch)
        label = category_id

        return image_data, label #, image_filename # use this for debug purposes

def preprocess(img_tar, anno_tar, anno_path, img_path, json_path, endstring):
    extract_tar(img_tar, img_path)
    extract_tar(anno_tar,  anno_path)
    voc_to_coco(anno_path, json_path, endstring)