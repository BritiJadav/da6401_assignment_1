# Deep_Learning_assignment_1
Q. 1 Log a W&B Table containing 5 sample images from each of the 10 classes in the dataset. Identify any classes that look visually similar in their raw form. How might this visual similarity impact your model?

Ans : Though MNIST digits are distinct, some digits looks similar due to handwriting variations.

Common visually similar pairs:

1.	1 and 7 - Some 7s look very similar to 1.
2.	3 and 5
3.	4 and 9
4.	0 and 6
5.	8 and 3
   
Impact of Visual Similarity on Model Performance

1.	Higher Misclassification Between Similar Digits -	The model may confuse 1 and 7, 3 and 5, etc.
2.	Overlapping Feature Representations -	Similar handwritten shapes produce similar pixel distributions.
3.	Decision Boundary Complexity - The model must learn highly non-linear boundaries to separate similar digits. If the network is shallow, it struggles.
4.	Effect on Confusion Matrix - Most errors will occur between visually similar classes. Confusion matrix will show strong off-diagonal values for those pairs.


 Q. 3 Compare the convergence rates of all 4 optimizers using the same architecture (3 hidden layers, 128 neurons each, ReLU activation). Which optimizer minimized the loss fastest in the first 5 epochs? Theoretically, why does RMSProp often outperform standard SGD on image classification?
 
Ans : RMSProp optimizer minimize the loss fastest in the first 5 epochs.

Image datasets like MNIST:

1.	High dimensional (784 inputs)
2.	Deep networks
3.	Many parameters
4.	Sparse gradients in ReLU
5.	Noisy gradients

RMSProp helps because it:

1. Normalizes gradient magnitude
2. Stabilizes updates
3. Prevents oscillations in steep directions
4. Accelerates convergence in early epochs.

Q. 4 Fix the optimizer to RMSProp and compare Sigmoid and ReLU for different network configurations. Log the gradient norms for the first hidden layer. Do you observe the vanishing gradient problem with Sigmoid? Provide a plot to support your observation

Ans : Using RMSProp optimizer, we compared Sigmoid and ReLU activations across different network configurations. The plot of gradient norm versus epoch showed that the gradients in the Sigmoid network rapidly decreased towards zero, indicating the presence of the vanishing gradient problem. In contrast, the ReLU network maintained significantly larger gradient norms across epochs.

