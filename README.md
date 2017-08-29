I'm trying the training process of MTCNN only for O-Net, not yet reached the author's precision. Any advice is welcome.

My label list is as follows:  
48/negative/0.jpg 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  
48/positive/0.jpg 1 0.055859 0.027310 -0.052455 0.114732 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  
48/part/0.jpg -1 0.059258 -0.119191 0.208781 0.282666 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  
48/landmark/0.jpg -1 -1 -1 -1 -1 0.266432 0.575103 0.387556 0.379742 0.661062 0.443078 0.411820 0.697048 0.783006 0.740027

Negtives and positives are for face classification tasks, positives and part faces are used for bounding box regression, and landmark faces are used for facial landmark localization. You should add the "EuclideanLossX" layer (folder layer, only GPU version) to your original caffe for training using my label format.

Prepare training data (folder train/train_prepare), we use P-Net (author's model) and R-Net (author's model) to detect faces from WIDER FACE to collect positives, negatives and part faces while landmark faces are detected from CelebA:  
1: Download wider-face and celeba dataset.  
2: use gen_onet_widerface.py to generate negative, positive and part samples.  
3: use gen_onet_celeba.py to generate landmark samples.  
4: use get_label_list.sh to get subset in order to keep the ratio (3:1:1:2).  
5: you'd better use show_label.py to check if the label data are correct.  
6: use convert_data_2_hdf5.py to convert training samples to hdf5 files.  

Train (folder train):  
./train_onet.sh

Note:  
The training data generate python script is modified from [DuinoDu/mtcnn](https://github.com/DuinoDu/mtcnn), so I recommend using his demo.py for test.

We test our trained O-Net model on 300-W dataset, here is some samples results:  
![image](https://github.com/daikankan/mtcnn/test/124212_1.jpg)  
![image](https://github.com/daikankan/mtcnn/test/79378097_1.jpg)  
![image](https://github.com/daikankan/mtcnn/test/1051618982_1.jpg)  
![image](https://github.com/daikankan/mtcnn/test/5106695994_1.jpg)  
