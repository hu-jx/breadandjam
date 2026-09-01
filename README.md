## **Setup and installation instructions** 

To start running the script, first install the required dependencies:  
``` pip install -r requirements.txt ```

1. Terminal

In terminal, locate the root directory breadandjam and run the ``` main.py ``` script using the following CLI command:  
``` python ./src/main.py —image_dir <image_directory_path> ```

## **Problem motivation**

As image generation capabilities of LLMs improve, it has become harder to differentiate an AI-generated image from a real one, which poses serious implications, especially in the spread of disinformation on social media platforms. Furthermore, existing AI image detectors often struggle when the images have been slightly edited or compressed, which is often the case with images shared on social media. 

**Aim**

We aim to build a robust AI image detector, ISeeU, that maintains good performance and accuracy even under image transformations. This can help reduce fraud that occur due to AI images circulating, such as people falling for scams due to real-looking visuals that support the scams’ claims.

## **Approaches**

We trained the classifier model by first processing random images in our train dataset using AlbumentationsX, then we pass the images simultaneously through a CLIP (Contrastive Language Image-Pretraining) model and a frequency branch, where the features are then concatenated before being passed into a classification model to produce the final binary result.

**Preprocessing**  
**Augmentations**  
In order to simulate the augmentations often found on images retrieved from social media, we randomly preprocessed the images used in the train and a test\_robust set using the Albumentations Python library, where we utilised the ImageCompression, GaussianBlur, Downscale, GaussNoise, and ColorJitter functions. We chose to randomly augment 50% of the images. 

**Dual Data Alignment (DDA)-inspired preprocessing**  
Dual Data Alignment (DDA) helps to reduce a model’s reliance on specific signals which AI images possess. First, real and AI-generated images can differ in low-level signals such as compression history, frequency content, and decoder-related artifacts, which may cause the model to learn shortcuts instead of real generation patterns. 

As our dataset does not contain real and synthetic pairs of the same image, we adapted the original DDA approach. For pixel alignment, real training images are reconstructed using Stable Diffusion 1.4 VAE so that both classes share similar decoder-related characteristics. For frequency alignment, both real and AI-generated images are recompressed using the same fixed JPEG quality. 

These steps are only applied to the training set and the test set is left unchanged to reflect performance on real-world images.

## **Feature extraction** 

We passed the images simultaneously through 2 branches, CLIP to extract the wider semantic meaning of the image and a frequency branch to examine the finer details and irregularities that may be present. 

**CLIP** 

The CLIP Vision Transformer is a pretrained image backbone that helps align visual and text representations into a shared embedding space. Because of this shared embedding space, high-level semantics of images can be retrieved from the image pixels to allow for the classifier head to identify signals from such. 

We first process the raw images into RGB pixels and then apply the CLIP AutoProcessor that is imported from HuggingFace along with the transformer. This is to ensure that image pixels are fit for feature extraction by the CLIP Vision Transformer. The patch-level token embeddings are then extracted by the CLIP Vision Model, and are passed to a custom pooling class AttentionPool. AttentionPool helps to weigh the patch-level tokens before extracting a feature vector by relevance, to the classifier head for training. 

**Frequency branch**

Generative models inherently leave behind distinct mathematical anomalies in the frequency domain, typically caused by upsampling, convolution operations, or checkerboard artifacts, which is what our frequency branch is designed to detect. The images, after first being resized to standard dimensions of 224x224 pixels, are first put through a mini CNN to extract features. We then utilised the FFT (Fast Fourier Transform) to extract global high-frequency signal patterns and anomalies from these features. Since the vector values returned from CLIP are only from range \-1 to 1, and those returned from the frequency branch can have magnitudes in the thousands, it was crucial that we normalised the values of the frequency branch before concatenating the vectors together. 

To optimise training across epochs (since feature branches possess frozen parameters), raw images were processed and features were precomputed and extracted from the images in one go to create a cached dataset for train, validation, test and robust\_test datasets. In training, the classifier head along with the pooling class then trains on all the cached extracted features. 

## **Classification**

A linear layer was added on top of the frozen parameters in the CLIP Vision transformer as well as the frequency branch. 5 Epochs (loops of training on the train dataset) was used to allow the linear layer to learn the features better and adjust weights over time. 

The criterion used to train the linear layer was binary cross entropy loss, chosen for numerical stability. An optimiser was used to counter overfitting that might be found in extracted features. Gradients were accumulated (found from the loss function) and were used to adjust weights in the linear layer over dataset batches and epochs.

## **Results**

We ran our model on the first 1000 samples of each folder in the CIFAKE REAL and FAKE dataset. Validation and test data came from the in WildFake’s coco 2017 dataset and the DALLE3 Advanced dataset. Validation data included the first 200 samples of each while test data included the next 200 samples of each. 

A random seed of 42 was used to ensure replicability. 

**Robustness Evaluation Summary**

| Metric | Robust | Clean |
| :---- | :---- | :---- |
| ROC\_AUC | 0.8191 | 0.8127 |
| Accuracy  \= (TP \+ TN) / total | 0.7450 | 0.7275 |
| Recall \= TP / (TP \+ FN) | 0.7850 | 0.7850 |
| Precision \= TP / (TP \+ FP) | 0.7269 | 0.7040 |

**Analysis**  
Based on our robustness evaluation, we see that the model performs consistently under augmentations, with the average ROC\_AUC for robust and clean test datasets to be 0.8159, which suggests that the model possesses decent discriminative power in classifying images to be real or fake. 

Accuracy and precision is found to be higher in robust datasets than clean datasets as well. This could be due to the model picking up stronger signals that are more significantly found in augmented samples versus the clean samples which may be more similar to the fake images. 

**Limitations**

1. Significant skew towards predicting positive (i.e. predicting to be fake)   
   1. We found a significant overlap between the projected probability distributions of real and fake images, leading to a close-to-1 best probability threshold used [^1]: 0.99476 (5.s.f). In a bid to resolve this, a calibrator (LogisticRegression) was added. However, due to the very small float value differences between real and fake images' raw predicted probabilities, the calibrator becomes weak in discerning, leading to similar probability distributions around 0.5 instead. This means that predicted probabilities all congregate around 0.5
   2. This could be due to fake images having stronger and much easier-to-detect features, while real images were found to be more heterogenous with no unifying features to detect.
   3. With more more time, we could look into other calibrations methods. More importantly, with more time, we could inspect the reasons why the projected probability distributions of real and fake images are overlapping significantly and add the necessary additional signals such that the model is able to recognise between real and fake better. 
2. Runtime limitations   
   1. VAE reconstruction along with precomputing features may take a relatively long run time due to limited compute memory. 

**Development of idea**

1. Feature branches

When first brainstorming the solution, we initially planned to extract features by computing the ELA (Error Level Analysis) of each image, then using the ELA patterns to train a CNN. [^2] When computing ELA on an edited image, an irregular pattern with different intensities will appear, which would then be picked up by the CNN to perform the classification. However, ELA does not discriminate between different editing techniques, and there was a high false positive rate on compressed real images. Hence, we scraped the idea and used the recommended hybrid approach of a low-level frequency feature extractor along with a high-level CLIP semantics feature branch to train our classifier head. 

**Team member contributions**

| Name | Contribution |
| :---- | :---- |
| Jiaxin | Implemented the CLIP feature branch and fixed the initial low ROC_AUC with AttentionPool; Integrated the preprocessing, feature branches with linear classifier head; Added cache datasets |
| Fucheng | Made streamlit dashboard; Preprocessing DDA; Augment a random subset of the dataset |
| Arwen | Added frequency branch;  Formulated the machine learning pipeline; Constructed the README  |

[^1]:  The best probability threshold was derived using the Youden’s J statistic.

[^2]:  [https://www.mdpi.com/1424-8220/23/22/9037](https://www.mdpi.com/1424-8220/23/22/9037) 
