## SMDH
The source code of the **S**egmentation-**E**nhanced **D**eep  **M**ulti-**S**cale **H**ashing (TCDH) framework.

## Paper
**Segmentation-Enhanced Deep Multi-Scale Hashing for Chest X-Ray Image Retrieval**

Linmin Wang, Qianqian Wang, Xiaochuan Wang, Mingxia Liu

## Dataset
We used the following dataset:

-COVID-QU-Ex dataset (Can be downloaded [here](https://www.kaggle.com/datasets/andyczhao/covidx-cxr2?select=competition_test))
Note that this dataset is not static but constantly changes due to the new COVID images being added. The set of training, Validation, and test images data we used can be found in Train.txt, Val.txt and test.txt, respectively.

## Dependencies
**SMDH** needs the following dependencies:

- python 3.8.5
- PIL == 9.2.0
- torch == 1.13.0
- numpy == 1.23.3
- torchvision == 0.14.0
