import json
from sys import argv
import os
def coco_truncate_file_names(path_to_coco_json:str,keep_from_pos:int,keep_to_pos:int) -> None:
    """ This function is used to rename the files after annotation with cvat. Cvat is adding some numbers in front of each filename, those are removed by this function.

    Args:
        path_to_coco_json (str): Absolute path to the json in wich the labels are stored.
        keep_from_pos (int): The old file name will be trucated, only the positions between keep_from_pos to keep_to_po will be kept
        keep_to_pos (int): The old file name will be trucated, only the positions between keep_from_pos to keep_to_po will be kept
    """

    with open(path_to_coco_json,"r") as f:
        data = json.load(f)
    for image in data["images"]:
        image["file_name"] = image["file_name"][keep_from_pos:keep_to_pos]
    os.remove(path_to_coco_json)
    with open(path_to_coco_json, 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    coco_truncate_file_names(path_to_coco_json = argv[1],keep_from_pos=argv[2],keep_to_pos=argv[3])
