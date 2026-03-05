# Deep_Learning_assignment_1
Q. 1 Log a W&B Table containing 5 sample images from each of the 10 classes in the dataset. Identify any classes that look visually similar in their raw form. How might this visual similarity impact your model?

Ans : Though MNIST digits are distinct, some digits are visually similar due to handwriting variations.

Common visually similar pairs:

1.	1 and 7 - Some 7s look very similar to 1.
2.	3 and 5
3.	4 and 9
4.	0 and 6
5.	8 and 3
   
Impact of Visual Similarity on Model Performance

1.	Higher Misclassification Between Similar Digits -	The model may confuse 1 and 7, 3 and 5, etc.
2.	Overlapping Feature Representations -	Similar handwritten shapes produce similar pixel distributions.
3.	Decision Boundary Complexity - The model must learn highly non-linear boundaries to separate similar digits. If the network is shallow or undertrained, it struggles.
4.	Effect on Confusion Matrix - Most errors will occur between visually similar classes. Confusion matrix will show strong off-diagonal values for those pairs.

 https://wandb.ai/ma24m006-indian-institute-of-technology-madras/ASSIGNMENT_1/runs/715gue7v
