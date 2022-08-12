The first file is creating the logistic regression model with stochasitc gradient descent, we calculate the loss, wrt to w and b and each itteration
take a step size towards the minimum. After we have all of our thetas, X^T * Theta to get our prediction vector and pass that into our sigmoid function
four our outout. This file also contains plots of the train and set test error decrease as we use sgd.

The second file is what happens if we first pass our original X (n x m) matrix into a non linear function, in this case the funtion
we're passing it into is Cos(10x) + c, after we transform our original matrix, we pass it into the model from above, and our prediction accuracy improves.
