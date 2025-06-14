# Free-Window Depth

This is the official PyTorch implementation code for Free-Window Depth.

## Environment
```
conda create -n freewindow python=3.9
conda activate freewindow
pip install -r requirements.txt
```
## Dataset
You can prepare the datasets KITTI and NYUv2 according to [here](https://github.com/ShuweiShao/NDDepth), and then modify the data path in the config files to your dataset locations. For HyperSim dataset, you can download the dataset from [here](https://github.com/apple/ml-hypersim) or use the [script](download.py) we provide. Besides, you need to process the raw Hypersim dataset use the [script](hypersim_process/preprocess_hypersim.py) we provided. Our processing is based on [Marigold](https://github.com/prs-eth/Marigold/blob/main/script/depth/dataset_preprocess/hypersim/README.md).

## Training
First download the pretrained encoder backbone from [here](https://github.com/microsoft/Swin-Transformer), and then modify the pretrain path in the config files.

Training the NYU-v2 model:
```
python NBD/train.py configs/arguments_train_nyu.txt
```

Training the KITTI model:
```
python NBD/train.py configs/arguments_train_kittieigen.txt
```

Training the HyperSim model:
```
python NBD/train_hypersim.py configs/arguements_train_hypersim.txt
```

## Evaluation
Evaluate the NYUv2 model:
```
python NBD/test.py configs/arguments_eval_nyu.txt
```

Evaluate the KITTI model:
```
python NBD/test.py configs/arguments_eval_kittieigen.txt
```

You can refer to `NBD/test.py` and implement the test script for Hypersim dataset.
## Acknowledgements


