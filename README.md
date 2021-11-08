# OCSAN

This is the demo of OCSAN. 
You can use [OpenFace2](https://github.com/TadasBaltrusaitis/OpenFace) to cut the face from training or testing videos.

## Command
--train-dir   --test-dir   --fake-dir 

All the samples the network need should arrange as follows.
If you have same samples of obama, please arrange the samples folders like：
```
/XXX/obama
├─obama_1
│      frame_det_00_000001.png
│      ...
├─2obama
│      frame_det_00_xxxxxx.png
├─...
└─last_obama
│      frame_det_00_xxxxxx.png
```
The path '/XXX/obama' is the path which you can enter into the command. 
Please ensure that the subfolder contains the name of the person. 
The prefix 'frame_det_00_' for each picture is named by OpenFace.

--famous

If you want to train or test the person which is in the Dataset like '\_01_' , please ignore it. If your samples are about the real person like obama or trump, you should active it. 

Please see other command explanation by -h.

## Dataset

We provide the real samples of leagders which mentioned in our paper on the Internet. The url is [here](https://pan.baidu.com/s/1Y9PgnnchZfholu026S-NnA). Extract code: jbbo. Please cut it by Openface before use. For Setup 1, we use actors in [DeepfakeDetection](https://ai.googleblog.com/2019/09/contributing-data-to-deepfake-detection.html),For Setup 2, we use the people in [Celeb-DF](http://www.cs.albany.edu/~lsw/celeb-deepfakeforensics.html)

## Enhancement

We provide the method to enhance the face samples. You can make it by enhancement.py 
