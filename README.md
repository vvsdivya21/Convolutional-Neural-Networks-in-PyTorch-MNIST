Below is the CNN configuration that achieved above 98% accuracy in training, validation, and testing. Any deviations from this configuration were commented to explain the changes, as failing to include such explanations would result in losing marks for this section.

	•	The inputs to the first layer were 28x28x1 images (1 for greyscale), matching the model’s expected input.
	•	A convolutional layer was added with 28 filters of size 5x5x1, using the ReLU non-linearity, with a stride of 1 in both directions, and padding as needed.
	•	While PyTorch typically auto-initializes weights, weights here were initialized as Gaussian random variables with a mean of 0 and variance of 0.0025, and biases were set to a constant of 0.1, suited for ReLU non-linearities. This manual weight and bias initialization was demonstrated in the code, as it was a requirement for marking.
	•	A max-pooling layer with a pool size of 2x2 and a stride of 2 was then added.
	•	A second convolutional layer was added with 16 filters of size 5x5x28, using a stride of 1 in both directions and the ReLU non-linearity, without padding. The weights and biases in this layer were initialized similarly to those in the first convolution layer.
	•	Another max-pooling layer with a pool size of 2x2 and a stride of 2 followed.
	•	A Flatten layer was added: torch.flatten documentation.
	•	A fully connected (dense) layer with 1024 units was added. Each unit in the max-pool connected to these 1024 units, with the ReLU non-linearity applied to these units.
	•	A second dense layer with 128 units and the ReLU non-linearity was then added.
	•	A Dropout layer was included to reduce overfitting, with a rate of 0.2.
	•	Finally, a fully connected layer was added to obtain 10 output units, followed by a log_softmax activation function.

The configuration of the CNN model was verified using the pytorch-summary package.

The cross-entropy loss function, also known as the negative log likelihood loss in PyTorch, was used.

	•	Accuracy was defined as the fraction of data correctly classified.
	•	For training, the SGD optimizer was used with a learning rate set to 1e-2 and a momentum of 0.9.
	•	The training set was further split into train and validation sets, with 10% allocated for validation. The training and validation accuracy for each training epoch was recorded, without touching the test dataset.
	•	Additionally, printing the accuracy or loss every 100 batches was helpful to monitor the training progress, although this was optional.

 The model was trained and evaluated on the MNIST dataset, using minibatches of size 32. Approximately 50 training epochs were run, with initial tests using fewer epochs to confirm steady progress. Once it was confirmed that the code functioned correctly, the model was allowed to train for more iterations to fully complete training. After ensuring that the optimization was working effectively, the resulting model was run on the test data.

The model was confirmed to have fully converged before running on the test data, which was verified by examining the loss and accuracy curves.

The training and validation loss and accuracy curves were plotted, and the model’s accuracy on the test set was reported.

All plots included labeled axes and a title. When displaying more than one variable on the same plot, a legend was included as required.
