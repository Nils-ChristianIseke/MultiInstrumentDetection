import codecs
import PIL.Image
import io
import json
import sys
import os

def load_json(path_to_coco_labels:str):
    '''
    Parameters:
        path_to_coco_labels (string): Path to the COCO label file (.json)
    Returns:
        data: json object containing the labels
    '''
    with open(path_to_coco_labels) as json_file:
        data = json.load(json_file)
    return data

def encode_image_in_base64(image):
    '''
    Encodes image in base64 to make images storable in text format
    '''
    img_pil = PIL.Image.open(image)
    f = io.BytesIO()
    img_pil.save(f, format='PNG')
    data = f.getvalue()
    encData = codecs.encode(data, 'base64').decode()
    encData = encData.replace('\n', '')
    return encData


def coco_to_labelme(path_to_coco_annotations:str,goal_dir:str, save_image_inside_json:bool = False, labelme_version:str ="5.0.1")-> None:
    '''
    Converts annotations from COCO-Format to LabelMe Format, at the moment only support for boundingboxes
    Parameters:
        path_to_coco_annotations (string): Path to the COCO .json file 
        save_image_inside_json (boolean): If True image data will be encoded in base64 and saved in the labelme json-file
    '''

    category_dict = {}
    with open(path_to_coco_annotations) as json_file:
        data = json.load(json_file)
    for category in data["categories"]:
        category_dict[category["id"]] = category["name"]
    
    for image in data["images"]:
        image_id = image["id"]
        image_width =image["width"]
        image_height = image["height"]
        image_filename = image["file_name"]
        label_me_dict= {}        
        label_me_dict["version"] = labelme_version
        label_me_dict["flags"] = {}
        label_me_dict["shapes"] =[]
        label_me_dict["imagePath"]= image_filename
        if save_image_inside_json == True:
            encoded = encode_image_in_base64(image_filename)
            label_me_dict["imageData"] = encoded
        else:
            label_me_dict["imageData"] = None
        label_me_dict["imageHeight"] = image_height
        label_me_dict["imageWidth"] = image_width 
        
        for annotation in data["annotations"]:
            annotation_category = annotation["category_id"]
            label_name = category_dict[annotation_category] 
            annotation_boundingbox = annotation["bbox"]
            x_min = annotation_boundingbox[0]
            x_max = x_min + annotation_boundingbox[2]
            y_min = annotation_boundingbox[1]
            y_max = y_min + annotation_boundingbox[3]
        
            if annotation["image_id"] == image_id:
                label_me_dict["shapes"].append({"label":label_name,"line_color":"","fill_color":""
                ,"points":[[x_min,y_min], [x_max,y_max]],"shape_type":"rectangle","flags":{}})
        with open(os.path.join(goal_dir,image_filename[:-3]+"json"), 'w') as fp:
            json.dump(label_me_dict, fp)
            

if __name__ == "__main__":
    coco_to_labelme(sys.argv[1], sys.argv[2])