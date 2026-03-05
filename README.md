# Deep_Learning_assignment_1
Q. 1 Log a W&B Table containing 5 sample images from each of the 10 classes in the dataset. Identify any classes that look visually similar in their raw form. How might this visual similarity impact your model?
Ans : Though MNIST digits are distinct, some digits are visually similar due to handwriting variations.
Common visually similar pairs:
1.	1 and 7
o	Some 7s look very similar to 1.
2.	3 and 5
3.	4 and 9
4.	0 and 6
5.	8 and 3
Impact of Visual Similarity on Model Performance
1.	Higher Misclassification Between Similar Digits
o	The model may confuse 1 and 7, 3 and 5, etc.
2.	Overlapping Feature Representations
o	Similar handwritten shapes produce similar pixel distributions.
3.	Decision Boundary Complexity
o	The model must learn highly non-linear boundaries to separate similar digits.
o	If the network is shallow or undertrained, it struggles.
4.	Effect on Confusion Matrix
o	Most errors will occur between visually similar classes.
o	Confusion matrix will show strong off-diagonal values for those pairs.
