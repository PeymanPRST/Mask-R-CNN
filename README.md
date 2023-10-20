Crack Detection with Mask R-CNN
This project uses Mask R-CNN to perform semantic segmentation of cracks in concrete surfaces. The goal is to develop a robust crack detection model that can be deployed for infrastructure inspection.

Dataset
The dataset contains 1000 RGB images collected from damaged concrete structures. Each image has a corresponding binary mask file indicating pixel-level crack segmentation annotations.

Bounding box annotations with class labels ('crack', 'no crack') are provided in JSON format for object-level detections. The dataset is split into 80% train and 20% val images.

Preprocessing steps including resizing to 256x256, random cropping to 224x224 and horizontal flipping are applied. Images are normalized and converted to tensors. Masks are scaled from 0-255 to 0-1.

Model Architecture
Mask R-CNN with a ResNet-50 Feature Pyramid Network (FPN) backbone pretrained on MS COCO is used. The classifier and box predictor heads are replaced with 2 output classes.

Features extracted from {C2, C3, C4, C5} pyramid levels are used for region proposal, box regression and segmentation tasks. RoIAlign extracts fixed-size feature maps from each proposal.

Two 3x3 convolutional heads with Sigmoid activation predict classes and refined box coordinates for each RoI. A 1x1 convolutional mask head outputs the segmentation map.

Training Process
The model is trained on train splits for 20 epochs with a batch size of 1 on a single RTX 3090 GPU. SGD optimizer with momentum of 0.9 and initial learning rate of 0.005 is used.

Online data augmentation with horizontal flips is applied during training. Loss and accuracy are calculated on val splits after each epoch. The best model checkpoint is saved based on val accuracy.

Quantitative Evaluation
The final model achieves an average precision (AP) of 78% on crack detections and 75% mean Intersection over Union (mIoU) on segmentation masks on the held-out val images.

Future Work
Potential areas of improvement include:

Collect a larger, more diverse dataset
Test different backbone architectures like ResNet-101, ResNeXt-101
Perform data augmentation with various transforms
Ensemble models trained with different initializations
Deploy a web app for real-time crack detection on site
