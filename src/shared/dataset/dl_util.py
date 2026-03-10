import os
import tarfile
import json
import xml.etree.ElementTree as ET
import re
import shutil

def extract_tar(file_path, dest_path):
    """
    Extrahiert tar-Files

    Args:
        file_path (str)
        dest_path (str)
    """
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    with tarfile.open(file_path, "r") as tar:
        tar.extractall(path=dest_path)

def extract_tgz(file_path, dest_path):
    """
    Extrahiert targz-Files

    Args:
        file_path (str)
        dest_path (str)
    """
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=dest_path)
        
def voc_to_coco(voc_folder, output_json, string):
    """
    Konvertiert Pascal VOC annotationen zu einem COCO.json file

    Args:
        voc_folder (str): Ordner mit VOC annotationen (.xml Dateien)
        dest_path (str): Speicherziel des json files
    """
    images = []
    annotations = []
    categories = {}
    annotation_id = 1
    category_id = 1
    image_id = 1
    dropped_imgs=0

    # Regex für richtige XML-Dateien (FIRSTJxxxxxx.x+xxxxxx.xml) dies ist nur für den D1 datensatz relevant da es mehrere "gleiche" files gibt
    #pattern = re.compile(r"^FIRSTJ[0-9]{6}\.[0-9]\+[0-9]{6}\.xml$")
    pattern = re.compile(r"^FIRSTJ[0-9]{6}\.[0-9]\+[0-9]{6}(M|MR|R)?\.xml$")

    for xml_file in os.listdir(voc_folder):
        if pattern.match(xml_file):
            xml_path = os.path.join(voc_folder, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            objects = root.findall("object")

            # Überspringen, wenn mehr als ein Objekt vorhanden ist
            if len(objects) != 1:
                #print(len(objects), xml_path)
                dropped_imgs += 1
                continue
            

            filename = root.find("filename").text
            
            # Die folgende line ist auch spezifisch für D1 und muss sonst auskommentirert werden.
            filename = filename.rsplit(".", 1)[0] + string
            width = int(root.find("size/width").text)
            height = int(root.find("size/height").text)
            
            images.append({
                "id": image_id,
                "file_name": filename,
                "xml_path": xml_path,
                "width": width,
                "height": height
            })
            
            # Annotationen extrahieren
            for obj in root.findall("object"):
                class_name = obj.find("name").text

                if class_name not in categories:
                    categories[class_name] = category_id
                    category_id += 1
                
                category = categories[class_name]

                bndbox = obj.find("bndbox")
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)
                bbox = [xmin, ymin, xmax - xmin, ymax - ymin]  # COCO-Format

                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0
                })
                
            annotation_id += 1

            image_id += 1

    coco_json = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": v, "name": k} for k, v in categories.items()]
    }
    
    with open(output_json, "w") as f:
        json.dump(coco_json, f, indent=4)

    print(f"Dateien wurden zu {output_json} konvertiert! Es wurden {dropped_imgs} gedropped.")
    
    """# Kopiere das JSON-File nach /cluster/home/schlehan/
    destination_path = "/cluster/home/schlehan/" + os.path.basename(output_json)
    shutil.copy(output_json, destination_path)"""