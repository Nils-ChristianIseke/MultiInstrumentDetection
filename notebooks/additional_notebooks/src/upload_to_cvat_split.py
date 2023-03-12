import fiftyone as fo
import fiftyone.zoo as foz

def upload_to_cvat_split(path_to_images:str,path_to_labels:str,username_cvat,password_cvat, label_field:str,project_name:str="PDSM",split_size:int=100,url:str="http://localhost:8080"):
    """This function uploads images a local CVa

    Args:
        path_to_images (str): Absolute path to the images 
        path_to_labels (str): Absolute path to the labels
        username_cvat (_type_): Login Username for CVAT
        password_cvat (_type_): Login Password for CVAT
        label_field (str): 
        project_name (str, optional): Defines a project all annotation tasks are assigend to. Defaults to "PDSM".
        split_size (int, optional): Define how many images should be loaded to CVAT at once. Needed because data uploaded is limeted in CVAT for one API call. Defaults to 100.
    """
    
    dataset = fo.Dataset.from_dir(
                dataset_type=fo.types.COCODetectionDataset, data_path=path_to_images,labels_path= path_to_labels,name ="PDSM")
  
    anno_keys = []
    
    for i in range(int(len(dataset)/split_size)):
        anno_key = "example_%d" % i
        view = dataset.skip(i*split_size).limit(split_size)

        view.annotate(
                anno_key =anno_key,
                username=username_cvat,
                password=password_cvat,
                label_field = label_field,
                url=url,
                project_name=project_name
            )
        anno_keys.append(anno_key)

