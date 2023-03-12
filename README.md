# Multi-instrument detection for stereo-endoscopy in heart surgery
by Bastian Westerheide, Lisa Wimmer, Nina Stegmayer, Moritz Bednorz und Nils-Christian Iseke

---
### This project consist of two main tasks. 
1. Data annotation of stereo-endoscopy heart surgery images. 
2. Implementation of a [Faster R-CNN](https://arxiv.org/abs/1506.01497) and in perspective a Stereo Faster R-CNN for multi-instance as well as multi-class detection.

This repository is documenting the result of the PDSM-course (University of Heidelberg Summer 2022). It is mainly addressed to our fellow students, who are working on the same project.

---

## Annotation

### Annotation Format
<details><summary> COCO (recommended) </summary>

We used [COCO](https://cocodataset.org) as annotation format. We are also providing you with a script which is converting COCO-annotations to LabelMe annotations. (Take care: Because COCO is significantly more comprehensive compared to the annotation format of LabelMe, the conversion may result in data loss, e.g.: attributes are not converted). We had no problems with using COCO.
</details> 
 
### Annotation Tools

There are many annotation tools you can choose from. We made experiences with two of them. Comparing these two we would strongly recommend the use of CVAT.

<details><summary> labelMe (not recommended) </summary>

To us [LabelMe](https://github.com/wkentaro/labelme) was initially recommended as annotation tool. We had some trouble using it, including:

- Difficulties regarding the installation process
- It is not possible to run two instances at the same time
  - Hence side to side comparison of stereo view annotations is not possible
- teamwork is not possible
 </details> 
 
<details><summary> CVAT (recommended) </summary>  

A suitable alternative for us was [CVAT](https://github.com/openvinotoolkit/cvat). It can be used online or installed locally. (A running version to test [CVAT](https://cvat.org)). Due to data privacy reasons and data upload limitations, it is recommended to host CVAT yourself. Some Advantages of CVAT:

- Fast and modern UI
- Integration of Machine Learning and OpenCV to support annotations (e.g.: you can load a pretrained model to support you in creating more annotations)
- Tasks can be assigned to team members
- Easy to supervise the annotation process 

Some Disadvantages:

-  Not possible to access the (locally hosted) webapp if connected to the VPN (VPN is necessary to connect to the SDS)
  - Workaround: Make a copy of the data to your local machine
- CVAT is renaming the images by concatenating some numbers to each image name. 
 - Workaround: Omit the images when exporting from CVAT (use the original ones --> Also benefits the export speed). Then use the jupyter notebook postprocessing_CVAT.ipynb to remove the concatenated numbers from the file names.
  The recommended way to use CVAT is to host one instance every team member has access to. Maybe it is possible to host it inside the VPN, which would make access to the data easier?

</details> 

  
### Data and Model Analysis Tools
<details><summary> FiftyOne </summary>
  
[FiftyOne](https://voxel51.com/fiftyone) is a powerful OS-tool for data and model analysis. If you have access to raw data you can use it to choose which images to annotate (e.g.: based on an uniqueness measure). Since an interface to CVAT exists you can easily uploaded the selected images. We are providing a notebook where this process is implemented. As soon as a trained model exists you can use FiftyOne to evaluate the model. For more possibilities check out the [documentation of FiftyOne](https://voxel51.com/docs/fiftyone).
 
</details>

### Annotation Process
<details><summary> Label Annotations </summary>
In the dataset following instruments exist:  

- Nadelhalter  
- Atraum. Pinzette  
- Knotenschieber  
- Klappenschere  
- Nervhaken  

In many images a needle is present as well. We were told that the benefit of needle detection is not that high, so it was omitted from labeling.

We tried to ensure a high label quality, by going through the annotations at least three time. At first we split the data into five parts, each part was labeled by one team member independently. Afterwards one person did a quality check of all the annotations with fyftyone, trying to ensure that the annotations were correct and homogeneous. In a third annotation process every annotation was checked with CVAT. Although we worked with previously annotated data (we had access to the annotations of the previous semester, so basically it was the fourth iteration of labeling) many annotation errors were found in our last annotation iteration.
To further improve the quality, we would recommend to check the labels again.

</details>

<details><summary> Annotation rules </summary>

- Bounding boxes should be as tight as possible 
- Set `occluded` tag if the instrument is occluded
- Use only one bounding box per instrument (Do not split if parts of the instrument are occluded)
- Only frame the actually visible part with a bounding box
- If the instrument is not clearly recognisable from the frames provided, it may help to look at the frames before and after. In the first step, this should be done on the basis of the frames provided. If this doesn't work, the original videos can be used (therefore you can contact your supervisor).   
</details>

<details><summary> Attribute Annotations </summary>

While checking the annotations more attributes where added. Some of them are not double checked and should be handled with care.

- occluded: True, if anything was occluding the view of the instrument
- clearly_classified: False, if the annotator was not sure about the class (Not double checked)
- blurry: True, if the instrument is blurry (Not double checked)

The motivation behind this was to provide the model with more information. This would need a custom implementation, since the used model is not equipped to handle these attributes.
</details>

---

## Model implementation

### Pipelines

<details><summary><strong>notebooks/model_implementations/Pytorch_Mono_Stereo_Faster_R_CNN.ipynb</strong> (click to expand)</h2></summary>

This notebook contains an implementation of a monovision and stereovision Faster R-CNN using our [custom COCO dataset](https://medium.com/fullstackai/how-to-train-an-object-detector-with-your-own-coco-dataset-in-pytorch-319e7090da5).  
</details>
<br/>



<details><summary><strong>notebooks/model_implementations/MMDetection_Mono_Faster_R_CNN.ipynb</strong> (click to expand)</h2></summary>

This notebook contains an implementation of a monovision Faster R-CNN based on a MMDetection [Tutorial](https://github.com/open-mmlab/mmdetection/blob/master/demo/MMDet_Tutorial.ipynb) using a custom dataset, configured train and test pipeline (augmentation) as well as evaluation. The complete notebook can be accessed via the following [LINK](https://colab.research.google.com/drive/1uuNPcOvQQBeg_Yd19bOveiGMEmyyjOY6?usp=sharing). Furthermore, a trained model, corresponding logs and data is provided via the following [LINK](https://drive.google.com/drive/folders/1mMuyMhRdz5PUlIBhdZygsuZyTu2EJ4Tn?usp=sharing)
</details>



### Faster R-CNN
<img align="right" width="25%" height="25%" src="https://production-media.paperswithcode.com/models/FASTER-RCNN_8CxhTTJ.png" alt="Faster R-CNN"/>
 
- Pretrained Faster R-CNN
- Backbone: ResNet50
- Mono: 3 Input Channels
- Stereo: 6 Input Channels
- Output
  - Bounding boxes
  - Labels
  - Confidence score
- Postprocessing: Non-Maximum-Suppression
 
---

## BwVisu
When using BwVisu there are some difficulties. It is still a little bit tricky to get some things running, but provides the desperately needed computational power in combination with data storage. On a helpful note: when using pytorch it is definitely worth checking the cuda versions running on BwVisu, so you don't run into issues with incompatibility. Installing python modules is generally easier done directly in a notebook (pip seems to work better than conda), opposed to installing via console. This might have changed in the meantime with BwVisu getting updates.

---

## SDS
To use the SDS in a running BwVisu instance, `kinit` must be executed in the terminal. Otherwise the access to the SDS is not possible. In some cases the directories did not show in the file explorer. This can be fixed by accessing them via terminal.
Unfortunately, our group members had only write access to the self-created directories / files during the whole semester. This makes collaborating in the same notebooks difficult. Consider a stronger integration of GitHub into the working process.

The annotations can be found in the directory: data_preprocessed:
- data_coco: contains for each video and view a seperated .json file (Format COCO)
- data_coco_all_in_one_json: contains 1 .json in which all annotations are stored (Format COCO)
- data_labelme: contains for each video and view a seperated .json file (Format Labelme)
- toy_data_rotated_bounding_box: contains toy data from one view to test rotated boundingboxes

---

## Where to find ...
### Data
<details><summary>(click to expand)</summary>

**Raw Data:**  
*Training data:*  
SDS: `/sd22A004/guest/dataset/instrument_detection_dataset_raw/`  
*Test data:*  
SDS: `/sd22a004/guest/dataset/instrument_detection_test_dataset/` 

**Processed data**  
*Final version used in the model:*  
SDS: `/sd22A004/guest/data_preprocessed/data_coco/`  
*Test data (needed a specific structure):*  
SDS: `/sd22A004/guest/data_preprocessed/data_testset/`  

**Notebooks/Tools used for data analysis:**     
GitHub: `wft_ss22_multi_instrument_detection/notebooks/additional_notebooks/datasets_json_plots.ipynb`   
GitHub: `wft_ss22_multi_instrument_detection/notebooks/additional_notebooks/datasets_plots_fiftyone.ipynb`  
</details>

### Model
<details><summary>(click to expand)</summary>

**Mono, stereo model implementation notebooks with info about hyperparams/config used:**   
SDS: `/sd22a004/guest/object_detection/model_implementations/Pytorch_Mono_Stereo_Faster_R_CNN.ipynb`  
GitHub: `wft_ss22_multi_instrument_detection/notebooks/model_implementations/Pytorch_Mono_Stereo_Faster_R_CNN.ipynb`   
Hyperparameters:  
- HEIGHT =  256 (height of input image)
- WIDTH = 256 (width of input image)
- BATCH_SIZE = 8
- NUM_WORKERS = 8 (for multiprocessing)
- NUM_CLASSES = 6 (5 classes plus one for the background) 
- NUM_EPOCHS = 100
- NUM_CHANNELS = 6 (Number of input channels for the ResNet50 backbone. Used for the stereo model)

**Pre-trained model weights with info on which one was used for reporting:**  
GitHub: `wft_ss22_multi_instrument_detection/trained_models/`  
--> for reporting: `pretrained_mono_100epochs_fold_1.pth` and `pretrained_stereo_100epochs_fold_1.pth`  
</details>

### Model output
<details><summary>(click to expand)</summary>

**Predictions from model:**  
SDS: `/sd22A004/guest/predictions/`  
GitHub: `wft_ss22_multi_instrument_detection/predictions/`  

**Post-processing code: Non-Maximum-Supression**  
SDS: `/sd22a004/guest/object_detection/model_implementations/Pytorch_Mono_Stereo_Faster_R_CNN.ipynb`  
GitHub: `wft_ss22_multi_instrument_detection/notebooks/model_implementations/Pytorch_Mono_Stereo_Faster_R_CNN.ipynb`  
--> NMS is used in the evaluate function  
</details>

### Evaluation
<details><summary>(click to expand)</summary>

**computation of metrics, scores, graphs generated:**  
*Hota:*  
SDS: `/sd22a004/guest/object_detection/evaluation/eval_hota_bbox.ipynb`  
GitHub: `wft_ss22_multi_instrument_detection/notebooks/evaluation/eval_hota_bbox.ipynb`  

*Mean average precision:*  
SDS: `/sd22a004/guest/object_detection/model_implementations/Pytorch_Mono_Stereo_Faster_R_CNN.ipynb`  
GitHub: `wft_ss22_multi_instrument_detection/notebooks/model_implementations/Pytorch_Mono_Stereo_Faster_R_CNN.ipynb`  
--> in chapter "Additional Content for future implementations"  

**Display images with ground truth and predictions:**  
SDS: `/sd22a004/guest/object_detection/evaluation/show_predictions.ipynb`  
GitHub: `wft_ss22_multi_instrument_detection/notebooks/evaluation/show_predictions.ipynb`  

**Create video from predictions:**    
SDS: `/sd22a004/guest/object_detection/evaluation/create_video.ipynb`  
GitHub: `wft_ss22_multi_instrument_detection/notebooks/evaluation/create_video.ipynb`  
</details>

### Other documentation 
<details><summary>(click to expand)</summary>

**Mid-term presentation:**   
GitHub: `wft_ss22_multi_instrument_detection/documentation/PDSM_midterm_presentation.pdf`   
**Final presentation:**   
GitHub: `wft_ss22_multi_instrument_detection/documentation/PDSM_final_presentation.pdf`  
**Details about the annotation process:**  
In this README under the section [Annotation process](#annotation-process)  
**List of images with difficulties in the annotation process:**  
GitHub: `wft_ss22_multi_instrument_detection/documentation/annotations_issues.md`  
**Video made from predictions:**  
SDS: `/sd22A004/guest/object_detection/predicted_test_video.avi`  
GitHub: `wft_ss22_multi_instrument_detection/documentation/predicted_test_video.avi`  


</details>

---

## Provided Scripts / Additional Notebooks
We provide a bunch of scripts to handle everything we needed during the project. These are located in `notebooks/additional_notebooks/`

<details><summary> Notebooks (click to expand)</summary>
 
`coco_to_labelme.ipynb`  --> used to convert COCO annotation format into labelMe format.

`datasets_json_plots.ipynb` --> Used to creates insights on the labels (instrument distribution,...), information is stored in pandas Dataframes. 

`datasets_plots_fiftyone.ipynb` --> Used to creates insights on the labels (instrument distribution,...), information is stored in fyftyone datasets

`json_field_deletion.ipynb` --> Used to delete the "label_id" in all annotations. That was necessary due to some bugs regarding the upload process to CVAT, maybe those are resolved in the meantime

`labelme_to_coco.ipynb` --> Used to convert the labelMe annotation format into COCO format.

`postprocessing_CVAT.ipynb` --> CVAT in concatenating some numbers to the beginning of each image, notebook is used to remove that.
 
`upload_to_cvat.ipynb` --> Uploads images and annotations to CVAT.

`evaluate_detections.ipynb` --> Copy of a FiftyOne Tutorial.

`exploring_image_uniqueness.ipynb` --> Not used, because no raw data was provided. Use it to find and remove near-duplicate images in your dataset and to recommend the most unique samples in your data.  (Based on a [FiftyOne Tutorial](https://voxel51.com/docs/fiftyone/tutorials/uniqueness.html)) 

`finding_detecion_errors.ipynb` --> Not used, because no trained model was provided. Use it to find mistakes in your annotations.
(Based on a [FiftyOne Tutorial](https://voxel51.com/docs/fiftyone/tutorials/detection_mistakes.html)).annotations. 

</details>


## Where to find ...
<details><summary> Data </summary>  
 
**Raw Data:**     
SDS: '\sd22A004\guest\dataset\instrument_detection_dataset_raw'  

**Processed data -- intermediate + final version used in the model:**    
SDS: '\sd22A004\guest\data_preprocessed\data_coco'  

**Notebooks/Tools used for data analysis:**     
--> GitHub: 'wft_ss22_multi_instrument_detection/additional_notebooks/datasets_json_plots.ipynb'   
--> GitHub: 'wft_ss22_multi_instrument_detection/additional_notebooks/datasets_plots_fiftyone.ipynb'  
</details> 
 
<details><summary> Model </summary>

**Mono, stereo model implementation notebooks with info about hyperparams/config used:**   
--> GitHub: 'wft_ss22_multi_instrument_detection/object_detection/WFT_Endo.ipynb'   
--> hyperparameters:  
- HEIGHT =  256 (height of input image)
- WIDTH = 256 (width of input image)
- BATCH_SIZE = 8
- NUM_WORKERS = 8
- NUM_CLASSES = 6
- NUM_EPOCHS = 100
- NUM_CHANNELS = 6 ( This is only used for the stereo model)

**Pre-trained model weights with info on which one was used for reporting:**   
SDS: '\sd22A004\guest\object_detection\model_backups'  
--> for reporting: 'pretrained_mono_100epochs_fold_1.pth' and 'pretrained_stereo_100epochs_fold_1.pth'  
</details> 

<details><summary> Model output </summary>

**Predictions form model - where are they stored:**  
SDS: '\sd22A004\guest\object_detection\predictions'

**Post-processing code: NMS**  
GitHub: 'wft_ss22_multi_instrument_detection/object_detection/WFT_Endo.ipynb' --> NMS is used in the evaluate function  
 </details> 
 
<details><summary> Evaluation </summary>

**computation of metrics, scores, graphs generated:**  
--> hota: GitHub: 'wft_ss22_multi_instrument_detection/object_detection/metric_calculation/eval_hota_bbox.ipynb'  
--> mean average precision: GitHub: 'wft_ss22_multi_instrument_detection/object_detection/WFT_Endo.ipynb' in chapter "Additional    Content for future implementations"

**prediction video:**    
SDS: '\sd22A004\guest\object_detection\predicted_test_video.avi'
 </details> 
 
<details><summary> Presentations </summary>

**Mid-term presentaion:**  
GitHub: 'wft_ss22_multi_instrument_detection/PDSM_midterm_presentation.pdf'   
**Final presentaion:**   
GitHub: 'wft_ss22_multi_instrument_detection/PDSM_final_presentation.pptx'
 </details> 
 
---

## Current issues
**Stereo model**  
While going through the notebooks we realized, that the dataloader implementation for stereo data had an issue in the augmentation process. The way it was implemented both sides were augmented differently. In the current version this is fixed by setting the same seed before augmenting the separate images.
This issue renders the results depicted in the presentation inaccurate. The results achieved with the errors fixed, have however no significant difference.

**Paths**  
By switching between bwVisu, local machines and a remote connection to the machine of our supervisor there might be some variations in the defined paths. Depending on the access to the SDS the paths have to be adjusted. Currently they should all be set to a value using the bwVisu structure, but it is possible that we might have overlooked one or another.
