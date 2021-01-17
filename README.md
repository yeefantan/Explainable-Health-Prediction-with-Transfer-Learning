# Explainable Health Prediction with Transfer Learning
This is a project studied on the sick and normal faces classification with Transfer Learning. Explainble AI technique is applied to find the attributes of each decision.

## Requirements
The code is written in Python and requires Tensorflow. You may install the requirements as follows:
```
pip install -r requirements.txt
```

## Dataset
In this study, the dataset composed of two categories as normal faces and faces with ill symptoms. The normal faces dataset is obtained from the UTKFace dataset [1]. The ill faces images are collected through online searching using keywords on Google. There are 1000 images is the normal faces category and 600 images in the sick faces category.

The data is then being pre-processed with Haar Cascade Classifier [2]. Data augmentation is applied to increase the dataset size.

## Experimental Results

## Model - VGGFace16
VGGFace-16 model [3] is used in extracting the image's features before training. The model is trained with the imagenet weights. After that, CNN [4] is designed and trained to classify the features. The accuracy obtained for training, validation, and testing are respectively 0.99, 0.98, and 0.98. The proposed method obtained a very high accuracy, but there is no clue in knowing the model's behaviour in making the decision. Hence, Explainable AI [5-7] techniques are applied to know the model's decision behaviour.

To run through the experiment, you may find it under 
```
/notebooks/transferlearning.ipynb
```
Alternatively, you can experiment it on 
```
transferlearning.py
```

## Explainable AI

We tested different Explainable AI techniques including XRAI [6], Integrated Gradients [7], and LIME [8].

### XRAI

### Integrated Gradients

### LIME
LIME can be applied to any machine learning model without knowing its underlying processing or internal representation. This is used to recognize the interpretable model on the interpretable attributes which are faithful to the regressor or classifier. 

To try the LIME execution, 
```
Go to /notebooks,
Run through LIME.ipynb
```

The results obtained from LIME is shown as below. 

|Original Image | LIME Explanations | Heatmap |
|---------------|-------------------|---------|
|![alt text](https://github.com/yeefan1999/Explainable-Health-Prediction-with-Transfer-Learning/blob/main/lime_explanations/1.jpg_explained.jpg "Logo Title Text 1")|![alt text](https://github.com/yeefan1999/Explainable-Health-Prediction-with-Transfer-Learning/blob/main/lime_explanations/1.jpg_explained.jpg "Logo Title Text 1")|![alt text](https://github.com/yeefan1999/Explainable-Health-Prediction-with-Transfer-Learning/blob/main/lime_explanations/1.jpg_explained.jpg "Logo Title Text 1")|
# References

[1] P. Viola, M. Jones, Rapid object detection using a boosted cascade of simple features, in: Proc. 2001 IEEE Comput. Soc. Conf. Comput. Vis. Pattern Recognit. CVPR 2001, IEEE Comput. Soc, Kauai, HI, USA, 2001: p. I-511-I–518. https://doi.org/10.1109/CVPR.2001.990517.

[2] Z. Zhang, Y. Song, H. Qi, Age Progression/Regression by Conditional Adversarial Autoencoder, ArXiv170208423 Cs. (2017). http://arxiv.org/abs/1702.08423 (accessed October 17, 2020).

[3] O.M. Parkhi, A. Vedaldi, A. Zisserman, Deep Face Recognition, in: Procedings Br. Mach. Vis. Conf. 2015, British Machine Vision Association, Swansea, 2015: p. 41.1-41.12. https://doi.org/10.5244/C.29.41

[4] Y. LeCun, Y. Bengio, G. Hinton, Deep learning, Nature. 521 (2015) 436–444. https://doi.org/10.1038/nature14539.

[5]	AI Explainability Whitepaper [Whitepaper], (n.d.). https://storage.googleapis.com/cloud-ai-whitepapers/AI%20Explainability%20Whitepaper.pdf.

[6]	A. Kapishnikov, T. Bolukbasi, F. Viégas, M. Terry, XRAI: Better Attributions Through Regions, ArXiv190602825 Cs Stat. (2019). http://arxiv.org/abs/1906.02825 (accessed October 20, 2020).

[7]	M. Sundararajan, A. Taly, Q. Yan, Axiomatic Attribution for Deep Networks, ArXiv170301365 Cs. (2017). http://arxiv.org/abs/1703.01365 (accessed October 20, 2020).

[8] M.T. Ribeiro, S. Singh, C. Guestrin, “Why Should I Trust You?”: Explaining the Predictions of Any Classifier, ArXiv160204938 Cs Stat. (2016). http://arxiv.org/abs/1602.04938 (accessed October 29, 2020).
