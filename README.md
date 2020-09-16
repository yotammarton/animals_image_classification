# Pet family and breed classification

Technion - Machine Learning and Optimization (097209)

Yotam Martin, Gal Goldstein


[<img src="https://i.imgur.com/EZpgCsf.png">](http://google.com.au/)

## Code directory

✓ sign means we build the code from scratch<br>
if no sign the code is from github repos

```bash
.
├── advanced_model_flat.py -------------------------------- train and inference
├── advanced_model_hierarchical.py
├── basic_model_baseline.py
├── basic_model_structured.py
├── crfrnn_keras
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
├── data_advanced_model.csv
├── data_advanced_model_linux.csv
├── data_basic_model.csv
├── data_basic_model_linux.csv
├── neural_structured_learning_adversarial_examples.py
├── neural_structured_learning_model.py
├── train_CRF-RNN
│   ├── LICENSE
│   ├── README.md
│   ├── TVG_CRFRNN_COCO_VOC_TEST_3_CLASSES.prototxt
│   ├── TVG_CRFRNN_COCO_VOC_TRAIN_3_CLASSES.prototxt ------
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
└── utils.py
```
