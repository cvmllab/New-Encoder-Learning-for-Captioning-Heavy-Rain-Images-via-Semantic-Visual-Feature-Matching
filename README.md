# New-Encoder-Learning-for-Captioning-Heavy-Rain-Images-via-Semantic-Visual-Feature-Matching

### 1.	Summary  
Image captioning generates text that describes scenes from input images. It has been developed for high-quality images taken in clear weather. However, in bad weather conditions, such as heavy rain, snow, and dense fog, the poor visibility owing to rain streaks, rain accumulation, and snowflakes causes a serious degradation of image quality. This hinders the extraction of useful visual features and results in deteriorated image captioning performance. To address practical issues, this study introduces a new encoder for captioning heavy rain images. The central idea is to transform output features extracted from heavy rain input images into semantic visual features associated with words and sentence context. To achieve this, a target encoder is initially trained in an encoder–decoder framework to associate visual features with semantic words. Subsequently, the objects in a heavy rain image are rendered visible by using an initial reconstruction subnetwork (IRS) based on a heavy rain model. The IRS is then combined with another semantic visual feature matching subnetwork (SVFMS) to match the output features of the IRS with the semantic visual features of the pretrained target encoder. The proposed encoder is based on the joint learning of the IRS and SVFMS. It is is trained in an end-to-end manner, and then connected to the pretrained decoder for image captioning. It is experimentally demonstrated that the proposed encoder can generate semantic visual features associated with words even from heavy rain images, thereby increasing the accuracy of the generated captions. 
 
### 2. Proposed network  
<img src="https://user-images.githubusercontent.com/73872706/136903184-e0360a64-1239-48e6-aac2-3b63cb23f606.png"  alt="Proposed network"></img>

### 3. Experimental results  
> #### 3.1 Synthetic heavy rain images  
> #### 3.2 Real heavy rain images  
 
### 4. Datasets  
> MSCOCO2014  
>	>Heavy rain,  [[here](https://drive.google.com/file/d/15_N7XM9PmiiljxsheSS688C11gmBdSvf/view?usp=sharing)]  
>	>IRS - A ,  [[here](https://drive.google.com/file/d/1SWPVlo0ACFw7azHyh7pKDhWnTi2g038N/view?usp=sharing)]   
>	>IRS - T,  [[here](https://drive.google.com/file/d/1iI1NGMYfS3rIas3DVRod6iRKEL_gZYeh/view?usp=sharing)]   
>	>IRS - S,  [[here](https://drive.google.com/file/d/17-k8My6b4v_m59jqm1VX3_ol5i3PYIpR/view?usp=sharing)]   
>	>Semantic Visual Feature, [[here](https://drive.google.com/file/d/19scHVYZfM3hhmSpLQcHutbal27P7zUwd/view?usp=sharing)]  

### 5.	Pretrained model  
> Proposed Model – (a)  
>	>Encoder, [[here](https://drive.google.com/file/d/1734-WdbjPiqsplQgDPBMcQ-jM1131qTD/view?usp=sharing)]  
>	>IRS, [[here](https://drive.google.com/file/d/1XeH90557xJCCJHNm0xHbj9ke0kSklCSz/view?usp=sharing)]  


> Proposed Model – (b)  
>	>Encoder, [[here](https://drive.google.com/file/d/1RI23XqY1hJyZ82Z4-sb-Kt_JIvhsQsbg/view?usp=sharing)]   


