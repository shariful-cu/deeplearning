#Deep Learning (DL) 

DL is a one kind of representation learning. For example, think about detecting objects from images. One node in a hidden layer (HL) may detect the rectangle pattern from the images and used that information for subsequence layers to detect a checkerboard inside the rectangle and later it uses this pattern to find something like a cat and so on. So, it goes like a representation learning. 



#Activation Function (AF) 

AF tells you how you want to adjust the output produced by each node in NN. For example,  



RELU that uses tanh (np.tanh()) function: always returns 0 if input is negative otherwise gives identical or same with the input of any positive numbers) 



#Loss Function (LF) 

LF computes the error at the final output nodes by comparing predicted final NN outputs with actual target outputs. For example, mean squared error (mse) computes a scalar error value (i.e., squared root of summing all errors at the output layer). 

Building a mode for regression problems (e.g., housing price prediction and stock prediction) mse is a good choice as a loss function 

Another loss function categorical_crossentropy is used for classification problem 

#Model Building Steps 

    Specify architecture 

    Compile 

    Fit 

    Predict 
