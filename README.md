# CMPC-Refseg
Code of our CVPR 2020 paper [*Referring Image Segmentation via Cross-Modal Progressive Comprehension*](https://openaccess.thecvf.com/content_CVPR_2020/papers/Huang_Referring_Image_Segmentation_via_Cross-Modal_Progressive_Comprehension_CVPR_2020_paper.pdf).

Shaofei Huang*, Tianrui Hui*, Si Liu, Guanbin Li, Yunchao Wei, Jizhong Han, Luoqi Liu, Bo Li (* Equal contribution)

## Interpretation of CMPC.

* (a) Input referring expression and image.

* (b) The model first perceives all the entities described in the expression based on entity words and attribute words, e.g., “man” and “white frisbee” (orange masks and blue outline).

* (c) After finding out all the candidate entities that may match with input expression, relational word “holding” can be further exploited to highlight the entity involved with the relationship (green arrow) and suppress the others which are not involved.

* (d) Benefiting from the relation-aware reasoning process, the referred entity is found as the final prediction (purple mask).
![interpretation](motivation.png)

## Experimental Results

We modify the way of feature concatenation in the end of CMPC module and achieve higher performances than the results reported in our paper.
New experimental results are summarized in the table bellow.
You can download our trained checkpoints to test on the four datasets. The link to the checkpoints is:
[Baidu Drive](https://pan.baidu.com/s/17TJDEiq5xA5ngN2jhsDQYA), pswd: 2miu.

| Method | UNC val | UNC testA | UNC testB | UNC+ val | UNC+ testA | UNC+ testB | G-Ref val | ReferIt test |
| :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
| STEP-ICCV19 \[1\] | 60.04 | 63.46 | 57.97 | 48.19 | 52.33 | 40.41| 46.40 | 64.13 |
| Ours-CVPR20 | 61.36 | 64.53 | 59.64 | 49.56 | 53.44 | 43.23 | 49.05 | 65.53 |
|Ours-Updated | **62.47** | **65.08** | **60.82** | **50.25** | **54.04** | **43.47** | **49.89** | **65.58** |

## Setup

We recommended the following dependencies.

* Python 2.7
* TensorFlow 1.5
* Numpy
* pydensecrf

This code is derived from [RRN](https://github.com/liruiyu/referseg_rrn) \[2\]. Please refer to it for more details of setup.

## Data Preparation
* Dataset Preprocessing

We conduct experiments on 4 datasets of referring image segmentation, including `UNC`, `UNC+`, `Gref` and `ReferIt`. After downloading these datasets, you can run the following commands for data preparation:
```
python build_batches.py -d Gref -t train
python build_batches.py -d Gref -t val
python build_batches.py -d unc -t train
python build_batches.py -d unc -t val
python build_batches.py -d unc -t testA
python build_batches.py -d unc -t testB
python build_batches.py -d unc+ -t train
python build_batches.py -d unc+ -t val
python build_batches.py -d unc+ -t testA
python build_batches.py -d unc+ -t testB
python build_batches.py -d referit -t trainval
python build_batches.py -d referit -t test
```

* Glove Embedding

Download `Gref_emb.npy` and `referit_emb.npy` and put them in `data/`. We provide download link for Glove Embedding here:
[Baidu Drive](https://pan.baidu.com/s/19f8CxT3lc_UyjCIIE_74FA), password: 2m28.


## Training
Train on UNC training set with:
```
python -u trainval_model.py -m train -d unc -t train -n CMPC_model -emb -f ckpts/unc/cmpc_model
```

## Testing
Test on UNC validation set with:
```
python -u trainval_model.py -m test -d unc -t val -n CMPC_model -i 700000 -c -emb -f ckpts/unc/cmpc_model
```

## CMPC for video referring segmentation
We release video version code for CMPC on A2D dataset under `CMPC_video/`.

## Reference
\[1\] Chen, Ding-Jie, et al. "See-through-text grouping for referring image segmentation." Proceedings of the IEEE International Conference on Computer Vision. 2019.

\[2\] Li, Ruiyu, et al. "Referring image segmentation via recurrent refinement networks." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.

## Citation
If our CMPC is useful to your research, please consider citing:
```
@inproceedings{huang2020referring,
  title={Referring Image Segmentation via Cross-Modal Progressive Comprehension},
  author={Huang, Shaofei and Hui, Tianrui and Liu, Si and Li, Guanbin and Wei, Yunchao and Han, Jizhong and Liu, Luoqi and Li, Bo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10488--10497},
  year={2020}
}
```
