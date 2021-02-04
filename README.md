# Description

Use VGG-16 and FCN-32 / FCN-8 models to do semantic segmentation on satellite images.

![image](https://github.com/weifanhaha/semantic-segmentation/blob/master/images/fcn.png)

## Model: Fully Convolutional Network - FCN32

![image](https://github.com/weifanhaha/semantic-segmentation/blob/master/images/semantic_segmentation.png)

Reference : [Long et al., “Fully Convolutional Networks for Semantic Segmentation”, CVPR 2015 (Best Paper)](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf)

# Usage

## Download Dataset

```
./get_dataset.sh
```

## Install packages

```
pip install -r requirements.txt
```

## Train

To train the semantic segmentation model, you can run the code by using the following command. please take care of the path of training data.

Train vgg16 + fcn32

```
python src/train.py
```

Train vgg16 + fcn8

```
python src/train_vggf8.py
```

## Predict

predict with vgg16 + fcn32

```
python3 p2/predict_bash.py --input_path $1/ --output_path $2/ --mode baseline
```

predict with vgg16 + fcn8

```
python3 p2/predict_bash.py --input_path $1/ --output_path $2/ --mode improved
```

### Evaluation

To evaluate your semantic segmentation model, you can run the provided evaluation script provided in the starter code by using the following command.

    python3 mean_iou_evaluate.py <--pred PredictionDir> <--labels GroundTruthDir>

-   `<PredictionDir>` should be the directory to your predicted semantic segmentation map
-   `<GroundTruthDir>` should be the directory of ground truth

Note that your predicted segmentation semantic map file should have the same filename as that of its corresponding ground truth label file (both of extension `.png`).

### Visualization

To visualization the ground truth or predicted semantic segmentation map in an image, you can run the provided visualization script provided in the starter code by using the following command.

    python3 viz_mask.py <--img_path xxxx_sat.jpg> <--seg_path xxxx_mask.png>

Note:
The code of fcn models are refered from [pytorch-fcn](https://github.com/wkentaro/pytorch-fcn)
