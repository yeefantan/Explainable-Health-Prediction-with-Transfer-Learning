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
VGGFace-16 model [3] is used in extracting the image's features before training. The model is trained with the imagenet weights. After that, CNN [4] is designed and trained to classify the features. The accuracy obtained for training, validation, and testing are respectively 0.99, 0.98, and 0.98. The proposed method obtained a very high accuracy, but there is no clue in knowing the model's behavior in making the decision. Hence, Explainble AI [5-7] techniques are experimented.

To run through the experiment, you may find it under /notebooks/transferlearning.ipynb. Alternatively, you can experiment it on transferlearning.py. 

## Explainable AI

# References

[1] P. Viola, M. Jones, Rapid object detection using a boosted cascade of simple features, in: Proc. 2001 IEEE Comput. Soc. Conf. Comput. Vis. Pattern Recognit. CVPR 2001, IEEE Comput. Soc, Kauai, HI, USA, 2001: p. I-511-I–518. https://doi.org/10.1109/CVPR.2001.990517.

[2] Z. Zhang, Y. Song, H. Qi, Age Progression/Regression by Conditional Adversarial Autoencoder, ArXiv170208423 Cs. (2017). http://arxiv.org/abs/1702.08423 (accessed October 17, 2020).

[3] O.M. Parkhi, A. Vedaldi, A. Zisserman, Deep Face Recognition, in: Procedings Br. Mach. Vis. Conf. 2015, British Machine Vision Association, Swansea, 2015: p. 41.1-41.12. https://doi.org/10.5244/C.29.41

[4] Y. LeCun, Y. Bengio, G. Hinton, Deep learning, Nature. 521 (2015) 436–444. https://doi.org/10.1038/nature14539.

[5]	AI Explainability Whitepaper [Whitepaper], (n.d.). https://storage.googleapis.com/cloud-ai-whitepapers/AI%20Explainability%20Whitepaper.pdf.

[6]	U. Pawar, D. O’Shea, S. Rea, R. O’Reilly, Explainable AI in Healthcare, (n.d.) 2.

[7]	A.B. Arrieta, Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI, Inf. Fusion. (2020) 34.
