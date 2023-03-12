The dataloader basically uses a text file that contains the path of the images to be loaded. The advantage of this is
that you could shuffle around different data splits, views, and temporal information with greater flexibility than
dealing with the data physically. Theoretically since the frame numbers are mentioned on the split file, you could
also pull up one frame before and after to experiment. 

A sample split file is provided in a folder. This splits the data for cross-validation. It was created with scipy such 
that the images from the sessions are mostly uniformly distributed. You could directly use this split or also generate
your own. The splitting function can also be provided to you if needed.

In this dataloader you may have to make some changes to the mask loading part where you have different folder names. An
example to instantiate the generator+dataloader in a model-ready format can be found in the notebook ```train.ipynb```