from torch.utils.data import Dataset, DataLoader
from PIL import Image
from astropy.io import fits
from shared.dataset.dl_util import *
from shared.dataset.scaling_functions import *

# Testpaths
#img_path = "./data/FITSImages/FITSImages/"
#coco_path = "./data/coco_annotation.json"

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
    def __init__(self, img_folder, annotation_file, order, gamma, transform_method, glob_min, glob_max, transform=None):
        self.img_folder = img_folder
        self.transform = transform
        self.order = order
        self.transform_method = transform_method
        self.gamma = gamma
        self.global_average_min = glob_min
        self.global_average_max =  glob_max
        # self.global_average_min = -0.0006134198629297316
        # self.global_average_max =  0.024582142010331154
        
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        self.image_id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

        self.annotations = []
        self.min = float('inf')
        self.max = float('-inf')
        dropout = 0
        for ann in data["annotations"]:
            image_id = ann["image_id"]
            category_id = ann["category_id"]
            
            # 50 % der Kategorie 1-Bilder ausschließen
            """if category_id == 1 and random.random() < 0.35:
                continue"""
            image_filename = self.image_id_to_filename[image_id]
            image_path = os.path.join(self.img_folder, image_filename)

                
            with fits.open(image_path) as hdul:
                image_data = hdul[0].data.astype(np.float32)

            image_data = np.nan_to_num(image_data, nan=0.0, posinf=1.0, neginf=0.0)

            current_min = image_data.min()
            current_max = image_data.max()

            if current_min < self.min:
                self.min = current_min
            if current_max > self.max:
                self.max = current_max
                
                
            self.annotations.append((image_id, category_id))
        self.category_mapping = {cat["id"]: cat["name"] for cat in data["categories"]}
        print(self.min, self.max,"droped:", dropout)

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
        
        """if image_data.max() >= 0.8:
            print(image_path)
        if image_filename == "FIRSTJ123048.4+122333.fits":
            print("Not rotated",image_data.max(), image_data.min())"""
        
        if self.order == True:
            if self.transform_method == "hep":
                image_data = hep_0(self.gamma[0], min = self.min, max = self.max, image_data = image_data)            
            elif self.transform_method == "z_scale_min_max":
                image_data = zscale_min_max(image_data)
            elif self.transform_method == "min_max":
                image_data = min_max(image_data)
            else:
                raise ValueError(f"Unknown Normalisaton: {self.order}")
            # Auf 3 Kanäle erweitern (z. B. für ResNet)
            image_data = np.repeat(image_data[:, :, np.newaxis], 3, axis=2)
        
        # Normalisierung für jeden Kanal
        elif self.order == False:
            
            # Auf 3 Kanäle erweitern (z. B. für ResNet)
            image_data = np.repeat(image_data[:, :, np.newaxis], 3, axis=2)
            
            if self.transform_method == "hep":
                image_data[:, :, 0] = hep_0(gamma = self.gamma[0], min=self.min, max=self.max, image_data = image_data[:, :, 0])
                image_data[:, :, 1] = hep_0(gamma = self.gamma[1], min=self.min, max=self.max, image_data = image_data[:, :, 1])
                image_data[:, :, 2] = hep_0(gamma = self.gamma[2], min=self.min, max=self.max, image_data = image_data[:, :, 2])
                
            elif self.transform_method == "hhz":
                image_data[:, :, 0] = hep_0(gamma = self.gamma[0], min=self.min, max=self.max, image_data = image_data[:, :, 0])
                image_data[:, :, 1] = hep_0(gamma = self.gamma[1], min=self.min, max=self.max, image_data = image_data[:, :, 1])
                image_data[:, :, 2] = zscale_min_max(image_data=image_data[:, :, 2])    
            
            elif self.transform_method == "hzz":
                image_data[:, :, 0] = hep_0(gamma=self.gamma[0], min=self.min, max=self.max, image_data = image_data[:, :, 0])
                image_data[:, :, 1] = zscale_min_max(image_data = image_data[:, :, 1])
                image_data[:, :, 2] = zscale_min_max(image_data=image_data[:, :, 2])
            
            elif self.transform_method == "hep_min_max":
                image_data[:, :, 0] = min_max(image_data=image_data[:,:,0])
                image_data[:, :, 1] = hep_0(gamma = self.gamma[1], min=self.min, max=self.max, image_data = image_data[:, :, 1])
                image_data[:, :, 2] = hep_0(gamma = self.gamma[2], min=self.min, max=self.max, image_data = image_data[:, :, 2])
            
            elif self.transform_method == "hep_min_zscale":
                image_data[:, :, 0] = min_max(image_data=image_data[:, :, 0])
                image_data[:, :, 1] = zscale_min_max(image_data = image_data[:, :, 1])
                image_data[:, :, 2] = hep_0(gamma = self.gamma[2], min=self.min, max=self.max, image_data = image_data[:, :, 2])
                
            else:
                raise ValueError(f"Unknown Normalisaton: {self.order}")
        else:
            print("No order on how to transform channels was given")
       
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