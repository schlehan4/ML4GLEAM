from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from shared.dataset.dataset import FitsSet
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

def show_test_images(dataloader, class_names, num_images=4, out_path=None):
    """
    Zeigt eine zufällige Auswahl von Testbildern mit ihren Labels an.

    Args:
        dataloader (dataset): Der dataset für den Test-Datensatz.
        class_names (dict): Ein Dictionary {class_id: class_name}.
        num_images (int): Anzahl der anzuzeigenden Bilder.
        out_path (str): Ziel falls man das Bild speichern möchte
    """
    images, labels = next(iter(dataloader))
    """indices = random.sample(range(len(images)), num_images)
    selected_images = images[indices]
    selected_labels = labels[indices]"""
    
    selected_images = images[:num_images]
    selected_labels = labels[:num_images]
    

    
    # Klassenverteilung aus dem gesamten dataset berechnen
    label_counts = {i: 0 for i in class_names.keys()}
    for _, batch_labels in dataloader:
        for label in batch_labels:
            label_counts[label.item()] += 1
            
    fig, axes = plt.subplots(2, num_images, figsize=(12, 6))

    for i in range(num_images):
        # Denormalisieren, um natürliche Farben wiederherzustellen
        #img = F.to_pil_image(selected_images[i])
        img = F.to_pil_image((selected_images[i] * 0.5) + 0.5)
        
        label_name = class_names[selected_labels[i].item()]
        
        axes[0,i].imshow(img)
        axes[0,i].axis("off")
        axes[0,i].set_title(label_name)
    
    
    
    labels, counts = zip(*label_counts.items())
    class_labels = [class_names[label] for label in labels]
    axes[1,0].bar(class_labels, counts, color='skyblue')
    for j in range(1, num_images):
        axes[1, j].axis("off")
    plt.tight_layout()
    
    plt.savefig(out_path) #falls man das bild aus debug gründen auf dem cluster speichern will auskommnetieren
    plt.show()
    


# Ordner für Bilder & Annotationen
# Hier müssen die Pfade für das Cluster angepasst werden! sollten noch keine Dateien im richtigen format existieren, müssen diese angepasst werden.
#img_folder = "data/D1_data/D1_images"
img_tar="C:/Users/Hannah/Desktop/AugmentedFITSImages.tar"
img_folder = "C:/Users/Hannah/Desktop/AugmentedFITSImages"
ann_tar = "C:/Users/Hannah/Desktop/AugementedAnnotations.tar"
ann_folder = "C:/Users/Hannah/Desktop/AugementedAnnotations"
annotation_file = "./data/coco_annotation.json"
out_path = "./data/out_plot_tests"

#preprocess(img_tar=img_tar, img_path=img_folder, anno_tar=ann_tar, anno_path=ann_folder, json_path=annotation_file,endstring=".fits" )
#voc_to_coco(voc_folder=ann_folder, output_json=annotation_file, string=".fits")
"""transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])"""

# Transformationen für das Modell (z. B. ResNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    #transforms.CenterCrop((112, 112)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.5], std=[0.5])  # Für resnet_fits-Bilder mit 1-Kanal
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Drei Kanäle normalisieren
])

train_dataset = FitsSet(f"{img_folder}", annotation_file, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

num_classes = len(train_dataset.category_mapping)
print(f"Number of classes: {num_classes}")
print(f"Total images loaded: {len(train_dataset)}")


class_names = {i: name for i, name in enumerate(train_dataset.category_mapping.values())}
print(class_names)
show_test_images(train_loader, class_names, num_images=6,out_path=out_path)


