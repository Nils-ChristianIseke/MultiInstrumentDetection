import sys
import fiftyone.core.utils as fou
import fiftyone as fo




def remove_images_duplications_from_dataset(dataset):
# Did not found any method to add also unlabeled images to the dataset, therefore I created the dataset form the labeled images
# than added all the images (labeled+ unlabeled) and then deleted the duplicates. --> Need to find a cleaner less error prune solution!!!
# https://voxel51.com/docs/fiftyone/recipes/image_deduplication.html
    for sample in dataset:
        sample["file_hash"] = fou.compute_filehash(sample.filepath)
        sample.save()
    dup_view = (dataset
        .sort_by("file_hash")
    )
    _dup_filehashes = set()
    for sample in dup_view:
        if sample.file_hash not in _dup_filehashes:
            _dup_filehashes.add(sample.file_hash)
            continue
        del dataset[sample.id]

def upload_to_cvat(path_to_images:str,path_to_labels:str, anno_key:str,username_cvat,password_cvat, label_field:str)->None:
    """Uploads images and labels to CVAT.

    Args:
        path_to_images (str): Absolut path to images
        path_to_labels (str): Absolut path to labels
        anno_key (str): Annotation key
        username_cvat (_type_): Username CVAT
        password_cvat (_type_): Password CVAT
        label_field (str): Label_field
    """
        
    dataset = fo.Dataset.from_dir(
                dataset_type=fo.types.COCODetectionDataset, data_path=path_to_images,labels_path= path_to_labels)

    dataset.add_images_dir(path_to_images)
    remove_images_duplications_from_dataset(dataset)
    dataset= dataset.sort_by("filepath")
    dataset.annotate(
        anno_key =anno_key,
        username=username_cvat,
        password=password_cvat,
        label_field = label_field,
        headers={"Referer": "https://cvat.org/api/auth/login"}
    )

def upload_to_cvat_local(path_to_images:str,path_to_labels:str, anno_key:str,username_cvat,password_cvat, label_field:str,project_name,url:str="http://localhost:8080"):
    """Uploads images and labels to CVAT.
     Args:
        path_to_images (str): Absolut path to images
        path_to_labels (str): Absolut path to labels
        anno_key (str): Annotation key
        username_cvat (_type_): Username CVAT
        password_cvat (_type_): Password CVAT
        label_field (str): Label_field
        url(str): The url where CVAT is hosted
    """
    
    print(path_to_images[-12:-7])
    dataset = fo.Dataset.from_dir(
            dataset_type=fo.types.COCODetectionDataset, data_path=path_to_images,labels_path= path_to_labels,name =path_to_images[-12:-7])
    dataset= dataset.sort_by("filepath")
    dataset.annotate(
        anno_key =anno_key,
        username=username_cvat,
        password=password_cvat,
        label_field = label_field,
        url=url,
        project_name=project_name
    )
        

if __name__ == "__main__":
    upload_to_cvat(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6])