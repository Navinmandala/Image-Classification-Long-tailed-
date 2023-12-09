# Notebook model contains Basic EfficientNET B3 model 

# Final_Notebook file contains Steps taken how to Handle  Overfitting on Long tailed Dataset and  Imbalanced Classes 


# STEPS TAKEN

# 1. Re-Weighting

Here our loss function is influenced by assigning relatively higher costs to examples from minority classes. We can use the re-weighting method from scikit-learn library to estimate class weights for unbalanced dataset with ‘balanced’ as a parameter which the class 
weights will be given by n_samples / (n_classes * np.bincount(y)).


# 2. Learning Rate Scheduler

Constant learning rate is the default learning rate schedule in any optimizer. It is tricky to choose the right learning rate in order to get the best optimization in the training phase. By experimenting with a range of learning rates in our example, lr=0.001 shows relatively good performance to start with. This can serve as a baseline for us to experiment with different learning rate strategies. The learning rate scheduler is to make the learning rate of optimizer adapt in a particular situation during the training phase. A learning rate scheduler relies on changes in loss function value to dictate whether the learning rate is decayed or not.

There are some types of learning rate scheduler.

#  Warm-up Learning Rate: 
In this method, we use to increase the learning rate gradually at the beginning of training phase, for example, we set the initial learning rate as small as possible and then multiply by a value to keep increase the learning rate until a certain number of epochs and the base learning rate is achieved. This idea is sometimes suitable to train for large-scale datasets to avoid "early-overfitting".

# Step Decay Learning Rate: 
this technique basically sets a different learning rate in a particular number of epochs. The values of the learning rate are defined at the beginning of training using a callback function.

# Cosine Decay Learning Rate: 
In this method, we use to decrease the learning rate gradually at the beginning of training phase, for example, we set the initial learning rate and then multiply by a cosine decay function to compute the decayed learning rate. Thus learning rate will keep decrease until a full range of training epochs.

# Adaptive Decay Learning Rate:
this technique reduce a learning rate adaptively according to the change of error in each epoch, if the error rate is not decreased in some epochs (we define the patience threshold by ourselves) then our learning rate will multiply by decay value to decrease the learning rate and prevent overfitting. In our case, we will use an adaptive learning rate scheduler to train the long-tailed dataset. (We can also use another type of learning rate scheduler and compare the results when doing the experiments).


# 3) Data Augmentation and Resampling
One of the basic approaches to deal with the imbalanced datasets is to do data augmentation and re-sampling. There are two types of re-sampling such as under-sampling when we removing the data from the majority class and over-sampling when we adding repetitive data to the minority class.

In this approach, We combine data augmentation and re-sampling technique by applying selective data augmentation to balance the dataset by re-sampling less frequent samples to adjust their amount in comparison with predominant samples. The augmentation is rotating the image by 10 degrees and change image brightness with a range of 0.2 to 1.0 (We also can do different data augmentation as well).


# 4. Change Loss Function
One of the common loss functions for solving the class imbalance problem is using Focal Loss. Instead of trying to reduce outliers or predictions where the model's prediction is far off from the truth, Focal Loss reduces the weight (or impact) the values it predicted correctly carry. The loss function is just a mathematical way of saying how far off a guess is from the real value of a data point. We also can do some research and experiment to use or create our own loss function, this study is called Metric Learning.




