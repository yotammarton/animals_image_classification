# Pet family and breed classification

Technion - Machine Learning and Optimization (097209)

Yotam Martin, Gal Goldstein


[<img src="https://i.imgur.com/EZpgCsf.png">](http://google.com)

## Code directory

✓ sign means we built the file from scratch<br>
if no sign the code is from github repos and adjusted by us

```bash
.
├── advanced_model_flat.py ----------------------------- ✓ train and inference flat models (breed)
├── advanced_model_hierarchical.py --------------------- ✓ train and inference hierarchical models (breed)
├── basic_model_baseline.py ---------------------------- ✓ train and inference KNN, SVM classifiers 
├── basic_model_structured.py -------------------------- ✓ inference structured FCN+CRF-RNN model
├── crfrnn_keras ---------------------------------------   github repo for training CRF-RNN 
│   ├── cpp
│   │   ├── Makefile
│   │   ├── high_dim_filter.cc
│   │   ├── modified_permutohedral.cc
│   │   └── modified_permutohedral.h
│   ├── crfrnn_layer.py
│   ├── crfrnn_model.py
│   ├── high_dim_filter_loader.py
│   ├── test_gradients.py
│   └── util.py
├── data_advanced_model.csv ---------------------------- ✓ paths to images for advanced part models (windows)
├── data_advanced_model_linux.csv ---------------------- ✓ paths to images for advanced part models (linux)
├── data_basic_model.csv ------------------------------- ✓ paths to images for basic part models (windows)
├── data_basic_model_linux.csv ------------------------- ✓ paths to images for basic part models (linux)
├── neural_structured_learning_adversarial_examples.py - ✓ generate examples for Figure 4 in report
├── neural_structured_learning_model.py ---------------- ✓ train and inference NSL models (creative part)
├── train_CRF-RNN --------------------------------------   github repo for training CRF-RNN 
│   ├── LICENSE
│   ├── README.md
│   ├── TVG_CRFRNN_COCO_VOC_TEST_3_CLASSES.prototxt
│   ├── TVG_CRFRNN_COCO_VOC_TRAIN_3_CLASSES.prototxt
│   ├── convert_labels.py
│   ├── crfasrnn.py
│   ├── data2lmdb.py
│   ├── filter_images.py
│   ├── loss_from_log.py
│   ├── resume_training.py
│   ├── solver.prototxt
│   ├── test_model.py
│   ├── train.py
│   └── utils.py
└── utils.py ------------------------------------------- ✓ aux funcs for augmentations and .csv files
```
