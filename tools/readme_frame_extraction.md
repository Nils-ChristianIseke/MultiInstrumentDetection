#### You can extract the frames of the video correctly by running the script in the repo as follows:

```
python extract_frames.py -v ~/Ch1_002_CH001_V.MP4 --successors 2
```

There are two ways to use this script:
1. Either you set the argument ```--successors``` to 1 or 2 and then you can get a concatenated image as the output.
2. If you don't want an _immediate_ successor but say 5 frames before and after, then set ```--sample_freq``` to 5 for
example, and then retain the frames that you need (from the frame mapping) and then delete the others.

**Note:** For aicm8 set the argument ```-s``` to 420 instead of 0, else you will get a lot of unwanted frames.
Please also make sure you install the required packages in your conda env before running this script
For aicm1 to aicm4, a frame grabber was used to grab the frames, so you can directly use these frames.
Also note that trying to compress the video in any way might introduce artefacts and compromise the stereo information,
so please only resize the images after you have extracted them and split them.
The extracted images have the left image at the top (image_02) and the right image at the bottom (image_03).
The split image helper function is also present in the ```extract_frames.py```, to then split the extracted frames.
Finally, bear in mind that since the image resolutions are huge and due to the advanced video encoding, generating the
images with successors might be slow.



