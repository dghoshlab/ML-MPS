![mps_bot](https://user-images.githubusercontent.com/63242780/181235055-1463e0e7-462f-4f0c-b4cb-a52d6a58cbc9.jpg)

# Introduction

Matrix product state (MPS) is a popular wave function ansatz in electronic structure theory. It aids the multi-dimensional coefficient tensor to be reduced in a much simpler form. Singular value decomposition (SVD) of a multi-dimensional coefficient tensor forms a MPS. In literature, MPS is usually represented as a chain form where each site is a smaller ranked tensor. Links between two tensors represent the entanglement or correlation. 

Here, the code generates a converged MPS by the supervised learning method. The data for learning can be generated from any ab initio method.The input and output comprise spin configurations on each site and the magnitude of CI coefficients, respectively. Here users can apply n-fold cross-validation to check the stability of the model. Finally, the converged model predicts a distribution of CI coefficients for some unknown datasets.

# Prerequisites :
1. Python 3.0+
2. PyTorch
3. TorchMPS 

# Contributors :
1. Sumanta Kumar Ghosh
2. Debashree Ghosh

# Compilation :
	a) First, install the TorchMPS software in workstation: git clone https://github.com/jemisjoky/TorchMPS.git
	b) Then install this code.

#How to run this code ?
Modify input arguments in "input.in" file. 
python3 mps_train.py input.in &

# Input arguments
1. M  		 :      INT
       			Bond dimention, represents the degree of correlation between two consecutive sites of MPS.
2. adaptive_mode : 	BOOL ( True/ False )
       			Whether the MPS adaptively chooses bond dimention while training.
3. periodic_bc 	 : 	BOOL ( True/ False )
			Whether the MPS will satisfy periodic boundary condition.
4. batch_size    : 	INT
			Batch size gives the amount of train data in a given batch while training of MPS. 
5. num_epochs    : 	INT
			The number of epochs/iterations for model training.
6. learn_rate    : 	FLOAT
			Learning rate of the model.
7. l2_reg        : 	FLOAT
			This is L2 regularization element in typical machine learning.
8. fold          : 	INT
			This gives how many fold cross validation a user want. 
9. inp_dim       : 	INT
			Dimension of input descriptor.
10. csv_column   : 	(INT a, INT b, INT c, INT d) 
			The range of input, output descriptor and determinant serial number should be given. Where, input ranges from 'a' to 'b' column of a csv file.
			c and d representing the determinant number and output descriptor respectively.
11. path         : 	STR
			Path of the directory where the csv data file is placed and ouput files will be generated.
12. TorchMPS_path:	STR
			Path of directory where TorchMPS is installed.
13. input_file   : 	STR
			Name of input data file in csv format. 

# Generated output files
After successful training, there will be three type of output file. In *Error* file there will be data of cost function with epochs for each fold training.
The converged model will be stored in *converged* file. Other output files will contain training and testing values for each fold of model optimization. 
