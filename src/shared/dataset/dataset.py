from torch.utils.data import Dataset, DataLoader
from PIL import Image
from astropy.io import fits
from shared.dataset.dl_util import *
from shared.dataset.scaling_functions import *

img_path = "./data/FITSImages/FITSImages/"
coco_path = "./data/coco_annotation.json"

class CocoLoader(Dataset):
    def __init__(self, img_folder, annotation_file, transform=None):
        self.img_folder = img_folder
        self.transform = transform
        
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        self.image_id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

        self.annotations = []
        for ann in data["annotations"]:
            image_id = ann["image_id"]
            category_id = ann["category_id"]
            self.annotations.append((image_id, category_id))

        self.category_mapping = {cat["id"]: cat["name"] for cat in data["categories"]}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_id, category_id = self.annotations[index]
        image_filename = self.image_id_to_filename[image_id]
        image_path = os.path.join(self.img_folder, image_filename)

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = category_id - 1
        return image, label
    
class FitsSet(Dataset):
    def __init__(self, img_folder, annotation_file, transform=None):
        self.img_folder = img_folder
        self.transform = transform
        
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        self.image_id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

        self.annotations = []
        for ann in data["annotations"]:
            image_id = ann["image_id"]
            category_id = ann["category_id"]
            
            # 50 % der Kategorie 1-Bilder ausschließen
            """if category_id == 1 and random.random() < 0.35:
                continue"""
            
            self.annotations.append((image_id, category_id))

        self.category_mapping = {cat["id"]: cat["name"] for cat in data["categories"]}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_id, category_id = self.annotations[index]
        image_filename = self.image_id_to_filename[image_id]
        
        image_path = os.path.join(self.img_folder, image_filename)

        # resnet_fits-Bild laden
        with fits.open(image_path) as hdul:
            image_data = hdul[0].data.astype(np.float32)

        # Ungültige Werte (NaN, Inf) behandeln
        image_data = np.nan_to_num(image_data, nan=0.0, posinf=1.0, neginf=0.0)
        
        #image_data = hep_zscale(1, image_data)
        
        # Auf 3 Kanäle erweitern (z. B. für ResNet)
        image_data = np.repeat(image_data[:, :, np.newaxis], 3, axis=2)
        
        # Normalisierung für jeden Kanal
        
        image_data[:, :, 0] = hep_zscale(gamma=2, image_data = image_data[:, :, 0])
        image_data[:, :, 1] = zscale_min_max(image_data = image_data[:, :, 1])
        image_data[:, :, 2] = zscale_min_max(image_data=image_data[:, :, 2])
       
        # In PIL-Bild umwandeln
        image_data = Image.fromarray((image_data * 255).astype(np.uint8))

        # Transformation anwenden (falls angegeben)
        if self.transform:
            image_data = self.transform(image_data)

        # Label anpassen (0-basiert für PyTorch)
        label = category_id - 1

        return image_data, label #, image_filename # use this for debug purposes

def preprocess(img_tar, anno_tar, anno_path, img_path, json_path, endstring):
    extract_tar(img_tar, img_path)
    extract_tar(anno_tar,  anno_path)
    voc_to_coco(anno_path, json_path, endstring)