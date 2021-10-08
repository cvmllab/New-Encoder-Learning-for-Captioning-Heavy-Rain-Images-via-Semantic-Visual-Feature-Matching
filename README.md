# New-Encoder-Learning-for-Captioning-Heavy-Rain-Images-via-Semantic-Visual-Feature-Matching

1.	Summary
Image captioning generates text that describes scenes from input images. It has been developed for high-quality images taken in clear weather. However, in bad weather conditions, such as heavy rain, snow, and dense fog, the poor visibility owing to rain streaks, rain accumulation, and snowflakes causes a serious degradation of image quality. This hinders the extraction of useful visual features and results in deteriorated image captioning performance. To address practical issues, this study introduces a new encoder for captioning heavy rain images. The central idea is to transform output features extracted from heavy rain input images into semantic visual features associated with words and sentence context. To achieve this, a target encoder is initially trained in an encoder–decoder framework to associate visual features with semantic words. Subsequently, the objects in a heavy rain image are rendered visible by using an initial reconstruction subnetwork (IRS) based on a heavy rain model. The IRS is then combined with another semantic visual feature matching subnetwork (SVFMS) to match the output features of the IRS with the semantic visual features of the pretrained target encoder. The proposed encoder is based on the joint learning of the IRS and SVFMS. It is is trained in an end-to-end manner, and then connected to the pretrained decoder for image captioning. It is experimentally demonstrated that the proposed encoder can generate semantic visual features associated with words even from heavy rain images, thereby increasing the accuracy of the generated captions. 
 
2. Proposed network
  

 
3. Experimental results
3.1 Synthetic heavy rain images
 
3.2 Real heavy rain images
 

4.	Datasets
MSCOCO2014 
	Heavy rain,  [here]
	IRS - A ,  [here]
	IRS - T,  [here]
	IRS - S,  [here]
	Semantic Visual Feature, [here]

5.	Pretrained model
Proposed Model – (a)
	Encoder, [here]
	IRS, [here]
Proposed Model – (b)
	Encoder, [here]


