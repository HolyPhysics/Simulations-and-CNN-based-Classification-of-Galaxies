import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
# print(torch.version)

# a little review for myself

batch_number, input_size, hidden_layer_size, output_size = 1000, 2, 50, 1

# input test tensors
x = torch.randn(batch_number, input_size)*3.1415 # This creates a "batch_number * input_size" tensor of dimension 2 (matrix)
# print(x.ndim)
# print(type(x).__name__)
# print(x.requires_grad)
y = ( x[:,0].sin() + x[:,1].cos() ) # this is still a vector with 100 columns
y = y.unsqueeze(1) # This adds an extra dimension at index 1, resulting in a "batch_number * output_size" tensor of dimension 2(matrix)
# print(y)
noise = torch.randn(batch_number, output_size)*1e-1
# print(noise)
y = y + noise # This makes learning the neural network work to learn the function y = sin(x_0) + cos(x_1)
# print(y)
plt.style.use("Solarize_Light2") # Sets a color theme for the background
figure, (ax_one, ax_two)= plt.subplots(1,2,figsize=(8.5,7.5))
ax_one.scatter(x.numpy()[:,0], x.numpy()[:,1], c=(y-noise).numpy()[:]) # Change is visible fomr the color grading
ax_one.set_xlim(-10,10)
ax_one.set_ylim(-10, 10)
ax_one.set_xlabel("Scatter diagram for actual function")

ax_two.scatter(x.numpy()[:,0], x.numpy()[:,1], c=y.numpy()[:])
ax_two.set_xlim(-10,10)
ax_two.set_ylim(-10, 10)
ax_two.set_xlabel("Scatter diagram for function with noise")
figure.tight_layout()
plt.show()



class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_layer_size, ouput_size) -> None:
        super().__init__()
        self.first_linear_layer = nn.Linear(input_size, hidden_layer_size)
        self.activation_function = nn.Sigmoid()
        self.second_linear_layer = nn.Linear(hidden_layer_size, ouput_size)

    def forward(self, X) -> list[float]: ## Forward must always be defined as it outlines the sequence/order the feedforward mechanism follows during the training of the neural network.
        output = self.first_linear_layer(X)
        output = self.activation_function(output) ## This ouput is different from the one above.
        y_pred = self.second_linear_layer(output)
        return y_pred

''' For ease of the optimization, I'm going to assume that the function the neural network 
will learn is convex at all value of the input so we can take negative gradient steps. 
To start the optimizartion and training, we initialize an instance of the model '''

model = TwoLayerNet(input_size, hidden_layer_size, output_size)

''' And we choose a loss function. For convenience, I'll go with the Mean Squared Error(MSE) function '''
loss_function = nn.MSELoss(reduction="sum")
learning_rate = 1e-3

# for values in range(batch_number+1):
#     '''Forward pass: Computes the predicted values of y by passing x to the model. It is
#     important to note that the Module object overrides the __call__ operator so one call 
#     them like functions. That is we can use: "model(x)". That is, one can pass tensors of input
#     data to the Module and it produces a tensor of output data 
#     '''
#     y_pred = model(x) # Recall that x is defined above

#     '''Now, we compute and print the loss. We just need to pass the predicted value and the true values
#     and the loss function returns a tensor containinf the loss.'''
#     loss = loss_function(y_pred, y) ## Again, recall y is defined above with noises
#     if values%100 == 0:
#         print(f'{values}-th iteration, loss: {loss.item()}') # loss.item() gives us the returned value for the loss(this is a single value for each iteration)

#     '''The next step is to zero all involved gradients before we proceed to run the backward pass
#     which wll compute the gradient for each iteration '''
#     model.zero_grad()

#     '''Backward pass: This will compute the gradient of the loss with respect to all the
#     learnable parameters of the model. Internally, the parameters of each Module aare stored
#     tensors with "requires_grad=True", so this call computes gradients for all learnable
#     parameters in the model. '''

#     loss.backward() # The gradient points in direction of steepest ascent. Easily follows from vector calculus

#     ''' Update weights using gradient descent. Each parameter is a tensor, so we can access its gradients
#     like we did before '''

#     with torch.no_grad(): 
#         ''' torch.no_grad() simply stops further tracking/computation of gradients this does not clear 
#         the gradients. Use optimizer.zero_grad() or model.zero_grad() to clear the gradients '''

#         for param in model.parameters():
#             param -= learning_rate*param.grad 
#             ''' We are able to take negative step because we have assumed 
#             the function is everywhere convex. Of course, this may not be the case. In such cases, we can use optimizers
#             like Adam, etc.'''


## For the case where the function is possibly not everywhere convex, it's much better to use the Adam optimzer 
# which can handle loss functions that occasionally increases. We do this as follows:



optimizer = optim.Adam(model.parameters(), lr=learning_rate)

''' The training loop is exactly the same except that the negative step is modified to 
optimizer.step() which takes a step in the direction opposite that of the gradient at that point. 
We also have a slight modification where we replace model.zero_grad() by optimizer.zero_grad()
 '''

for values in range(batch_number + 1):
    y_pred = model(x)
    loss = loss_function(y_pred, y)
    if values%100 == 0:
        print(f'{values}-th iteration, loss: {loss.item()}')
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()