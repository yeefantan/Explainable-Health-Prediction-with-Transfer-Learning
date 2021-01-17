# Explainable Health Prediction with Transfer Learning
This is a project studied on the sick and normal faces classification with Transfer Learning. Explainble AI technique is applied to find the attributes of each decision.

## Requirements
The code is written in Python and requires Tensorflow. You may install the requirements as follows:
{{box op="start" cssClass="boxed noteBox"}}
pip install -r requirements.txt

This is something important to keep in mind.
{{box op="end"}}

## Dataset

## Experimental Results

## Model - VGGFace16
VGGFace-16 model [1] is used in extracting the image's features before training. The model is trained with the imagenet weights. After that, CNN [2] is designed and trained to classify the features. The accuracy obtained for training, validation, and testing are respectively 0.99, 0.98, and 0.98. The proposed method obtained a very high accuracy, but there is no clue in knowing the model's behavior in making the decision. Hence, Explainble AI techniques are experimented.

To run through the experiment, you may find it under /notebooks/transferlearning.ipynb. Alternatively, you can experiment it on transferlearning.py. 

## Explainable AI

# References

[1] O.M. Parkhi, A. Vedaldi, A. Zisserman, Deep Face Recognition, in: Procedings Br. Mach. Vis. Conf. 2015, British Machine Vision Association, Swansea, 2015: p. 41.1-41.12. https://doi.org/10.5244/C.29.41

[2] Y. LeCun, Y. Bengio, G. Hinton, Deep learning, Nature. 521 (2015) 436–444. https://doi.org/10.1038/nature14539.

[3]	AI Explainability Whitepaper [Whitepaper], (n.d.). https://storage.googleapis.com/cloud-ai-whitepapers/AI%20Explainability%20Whitepaper.pdf.

[4]	U. Pawar, D. O’Shea, S. Rea, R. O’Reilly, Explainable AI in Healthcare, (n.d.) 2.

[5]	A.B. Arrieta, Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI, Inf. Fusion. (2020) 34.
