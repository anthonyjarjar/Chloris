# Chloris

Welcome to Chloris, a modern application designed to help users connect with the environment through technology. Our app leverages the power of Node.js and Expo Go to provide a seamless mobile experience.

## Getting Started

This guide will help you set up Chloris on your local machine for development and testing purposes. Follow these simple steps to get started.

### Prerequisites

Before you begin, ensure you have the following installed:
- Node.js (v14 or newer)
- npm (Node Package Manager)
- Expo CLI
- A code editor of your choice (e.g., VS Code)

### Directory Tree

- [**ChlorisApp**](https://github.com/anthonyjarjar/Chloris/tree/master/ChlorisApp)
  - [`App.js`](https://github.com/anthonyjarjar/Chloris/blob/master/ChlorisApp/App.js)
  - [`Home.js`](https://github.com/anthonyjarjar/Chloris/blob/master/ChlorisApp/Home.js)
  - [`ImageUpload.js`](https://github.com/anthonyjarjar/Chloris/blob/master/ChlorisApp/ImageUpload.js)
  - [`LocationPrediction.js`](https://github.com/anthonyjarjar/Chloris/blob/master/ChlorisApp/LocationPrediction.js)
  - [`app.json`](https://github.com/anthonyjarjar/Chloris/blob/master/ChlorisApp/app.json)
  - [**assets**](https://github.com/anthonyjarjar/Chloris/tree/master/ChlorisApp/assets)
    - [`adaptive-icon.png`](https://github.com/anthonyjarjar/Chloris/blob/master/ChlorisApp/assets/adaptive-icon.png)
    - [`favicon.png`](https://github.com/anthonyjarjar/Chloris/blob/master/ChlorisApp/assets/favicon.png)
    - [`icon.png`](https://github.com/anthonyjarjar/Chloris/blob/master/ChlorisApp/assets/icon.png)
    - [`splash.png`](https://github.com/anthonyjarjar/Chloris/blob/master/ChlorisApp/assets/splash.png)
  - [`babel.config.js`](https://github.com/anthonyjarjar/Chloris/blob/master/ChlorisApp/babel.config.js)
  - [`bird_names_dict.js`](https://github.com/anthonyjarjar/Chloris/blob/master/ChlorisApp/bird_names_dict.js)
  - [`package-lock.json`](https://github.com/anthonyjarjar/Chloris/blob/master/ChlorisApp/package-lock.json)
  - [`package.json`](https://github.com/anthonyjarjar/Chloris/blob/master/ChlorisApp/package.json)
- [`LICENSE`](https://github.com/anthonyjarjar/Chloris/blob/master/LICENSE)
- [`README.md`](https://github.com/anthonyjarjar/Chloris/blob/master/README.md)
- [**api**](https://github.com/anthonyjarjar/Chloris/tree/master/api)
  - [`app.py`](https://github.com/anthonyjarjar/Chloris/blob/master/api/app.py)
  - [**models**](https://github.com/anthonyjarjar/Chloris/tree/master/api/models)
    - [`__init__.py`](https://github.com/anthonyjarjar/Chloris/blob/master/api/models/__init__.py)
    - [`model.py`](https://github.com/anthonyjarjar/Chloris/blob/master/api/models/model.py)
    - [`one_class_svm_model.pkl`](https://github.com/anthonyjarjar/Chloris/blob/master/api/models/one_class_svm_model.pkl)
    - [`resnet_V5.pth`](https://github.com/anthonyjarjar/Chloris/blob/master/api/models/resnet_V5.pth)
    - [`testing_data.npz`](https://github.com/anthonyjarjar/Chloris/blob/master/api/models/testing_data.npz)
  - [`requirments.txt`](https://github.com/anthonyjarjar/Chloris/blob/master/api/requirments.txt)
  - [**src**](https://github.com/anthonyjarjar/Chloris/tree/master/api/src)
    - [`__init__.py`](https://github.com/anthonyjarjar/Chloris/blob/master/api/src/__init__.py)
    - [**jupyter notebooks**](https://github.com/anthonyjarjar/Chloris/tree/master/api/src/jupyter%20notebooks)
    - [`ocsvm.py`](https://github.com/anthonyjarjar/Chloris/blob/master/api/src/ocsvm.py)
    - [`speciescodes.py`](https://github.com/anthonyjarjar/Chloris/blob/master/api/src/speciescodes.py)
- [**documents**](https://github.com/anthonyjarjar/Chloris/tree/master/documents)
  - [`AnthonyJarjour_ProjectReport.pdf`](https://github.com/anthonyjarjar/Chloris/blob/master/documents/AnthonyJarjour_ProjectReport.pdf)
  - [`Chloris.pptx`](https://github.com/anthonyjarjar/Chloris/blob/master/documents/Chloris.pptx)
  - [`proposal.md`](https://github.com/anthonyjarjar/Chloris/blob/master/documents/proposal.md)


### Installation

1. **Clone the repository**

   Start by cloning the Chloris repository to your local machine. Use the following command in your terminal:

   ```bash
   https://github.com/anthonyjarjar/Chloris
   cd chloris
    ```

2. **Install dependencies**
    
    Navigate to the app directory and install the required packages
    
    ```bash
    cd /ChlorisApp
    npm install
    npm install -g expo-cli
    expo install
    ```
    
3. **Install Python Libraries**

    Navigate to the api directory and install required libraries
    
     ```bash
     cd /api
     pip install -r requirments.txt
      ```
      
4. **Launch the servers**

    To launch the expo app and the fastapi server, navigate to the respsective directories where [App.js](https://github.com/anthonyjarjar/Chloris/blob/main/ChlorisApp/App.js) and [app.py](https://github.com/anthonyjarjar/Chloris/blob/main/api/app.py) are and run the following commands:
    
    ```bash
     cd /api
     uvicorn app:app --host 192.***.*.* --port 8000 
      ```
      
    Make sure to use your local network, and change line 35 and 39 in 
    [ImageUpload.js](https://github.com/anthonyjarjar/Chloris/blob/main/ChlorisApp/ImageUpload.js) and [LocationPredictoin.js](https://github.com/anthonyjarjar/Chloris/blob/main/ChlorisApp/LocationPrediction.js)
    
    ```bash
     cd /ChlorisApp
     npx expo start 
      ```
**Now you're ready to go and checkout the app with the expogo mobile app!**
****
# Code Walkthrough

This walk through will only include the relevant sections for the relevant files that include concepts for AI and Machine learning

1. [image_classifier.ipynb](https://github.com/anthonyjarjar/Chloris/blob/main/api/src/jupyter%20notebooks/image_classifier.ipynb)
    ```python
    class BasicBlock(nn.Module):
        expansion = 1
    
        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
    
            #residual function
            self.residual_function = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, 
                kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
    
            self.shortcut = nn.Sequential()
    
            if stride != 1 or in_channels != BasicBlock.expansion * 
            out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, 
                    out_channels * BasicBlock.expansion, kernel_size=1, 
                    stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels * BasicBlock.expansion)
                )
    
        def forward(self, x):
            return nn.ReLU(inplace=True)(self.residual_function(x) + 
            self.shortcut(x))
    
    class BottleNeck(nn.Module):
    
        expansion = 4
        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            self.residual_function = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, stride=stride, 
                kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, 
                kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion),
            )
    
            self.shortcut = nn.Sequential()
    
            if stride != 1 or in_channels != out_channels * 
                BottleNeck.expansion:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * 
                    BottleNeck.expansion, stride=stride, kernel_size=1, 
                    bias=False),
                    nn.BatchNorm2d(out_channels * BottleNeck.expansion)
                )

        def forward(self, x):
            return nn.ReLU(inplace=True)(self.residual_function(x) + 
            self.shortcut(x))
    
    class ResNet(nn.Module):
    
        def __init__(self, block, num_block, num_classes=525):
            super().__init__()
    
            self.in_channels = 64
    
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True))
            # I use a different inputsize than the original paper
            # so conv2_x's stride is 1
            self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
            self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
            self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
            self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)
    
        def _make_layer(self, block, out_channels, num_blocks, stride):
    
    
            # I have num_block blocks per layer, the first block
            # could be 1 or 2, other blocks would always be 1
            strides = [stride] + [1] * (num_blocks - 1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_channels, out_channels, stride))
                self.in_channels = out_channels * block.expansion
    
            return nn.Sequential(*layers)
    
        def forward(self, x):
            output = self.conv1(x)
            output = self.conv2_x(output)
            output = self.conv3_x(output)
            output = self.conv4_x(output)
            output = self.conv5_x(output)
            output = self.avg_pool(output)
            output = output.view(output.size(0), -1)
            output = self.fc(output)
    
            return output
    
    def resnet50():
        return ResNet(BottleNeck, [3, 4, 6, 3])
    ```
> Please open this ipynb in a seperate window as syntax looks different in markdown file

## Classes and Functions

### `BasicBlock` Class

#### Description
- `BasicBlock` is a building block for a variant of the ResNet architecture, often used for smaller ResNet models like ResNet-18 and ResNet-34.
- This class inherits from `nn.Module`.

#### Attributes
- `expansion`: Static attribute used to control the expansion of the number of channels in the block. Set to 1 in `BasicBlock`.

#### Constructor (`__init__`)
- Parameters:
  - `in_channels`: Number of input channels to the block.
  - `out_channels`: Number of output channels from the convolutional layers before expansion.
  - `stride`: The stride size used for convolution, default is 1.
- The constructor sets up the residual function comprising two convolutional layers with Batch Normalization and ReLU activation.
- A shortcut connection is also defined, which is a convolutional layer that matches the dimensions and stride of the residual function if necessary (either the stride is not 1 or the number of input channels does not match the output channels after expansion).

#### `forward` Method
- Performs the forward pass of the block.
- It adds the output of the residual function to the shortcut and applies a ReLU activation function.

### `BottleNeck` Class

#### Description
- `BottleNeck` is a more complex building block for larger ResNet models like ResNet-50, ResNet-101, and ResNet-152.
- Inherits from `nn.Module`.

#### Attributes
- `expansion`: Static attribute set to 4, expanding the number of channels in the final layer of the block by this factor.

#### Constructor (`__init__`)
- Similar to `BasicBlock` but with three convolutional layers:
  - First and last convolutional layers have a kernel size of 1 (used for reducing and then expanding dimensions, respectively).
  - The middle convolutional layer has a kernel size of 3 and can have a stride greater than 1.
- Defines a shortcut connection similar to `BasicBlock`.

#### `forward` Method
- Functionality is similar to `BasicBlock`, combining the residual function's output with the shortcut's output using a ReLU activation.

### `ResNet` Class

#### Description
- Implements the ResNet architecture.

#### Constructor (`__init__`)
- Parameters:
  - `block`: Type of block to use (`BasicBlock` or `BottleNeck`).
  - `num_block`: List of integers defining the number of blocks in each of the four layers of the network.
  - `num_classes`: Number of classes for the output layer (default is 525).
- Sets up the initial convolutional layer (`conv1`) and then constructs each layer of the network using the `_make_layer` helper function.
- Includes an adaptive average pooling layer and a fully connected layer.

#### `_make_layer` Method
- Helper function to create a sequence of blocks for a layer of the network.
- Manages the number of blocks and the stride of the first block in each layer.

#### `forward` Method
- Defines the forward pass through the entire network.

### `resnet50` Function

#### Description
- Factory function that creates a ResNet model using the `BottleNeck` block configuration suitable for a ResNet-50 model.

### Execution Code

- Creates an instance of `ResNet` for a 50-layer model.
- Moves the model to a specified device (e.g., GPU).
- Prints out the model structure.

***
Now lets analyze the training loop 

```python
from functools import total_ordering
from torch.optim.lr_scheduler import ReduceLROnPlateau

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-5, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, cooldown=1, min_lr=3e-8, eps=1e-09)

num_epochs = 15
print(f"Training for {num_epochs} epochs...")

best_val_accuracy = 0
epochs_no_improve = 0
early_stop_threshold = 3

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for i, batch in enumerate(train_loader):
        inputs, labels = batch["pixel_values"], batch["label"]
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        loss = loss_fn(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels.to(device)).sum().item()
        total += labels.size(0)

    train_losses.append(train_loss / len(train_loader))
    train_accuracy = correct / total
    train_accuracies.append(train_accuracy)

    val_loss = 0.0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i, vdata in enumerate(val_loader):
            vinputs, vlabels = vdata["pixel_values"], vdata["label"]
            voutputs = model(vinputs.to(device))
            vloss = loss_fn(voutputs, vlabels.to(device))
            val_loss += vloss.item()
            _, predicted = torch.max(voutputs, 1)
            correct += (predicted == vlabels.to(device)).sum().item()
            total += vlabels.size(0)

    val_accuracy = correct / total
    val_accuracies.append(val_accuracy)

    val_losses.append(val_loss / len(val_loader))
    val_loss_epoch = val_loss/len(val_loader)

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve == early_stop_threshold:
        print("Early stopping triggered. Validation accuracy did not improve for 10 consecutive epochs.")
        break

    scheduler.step(val_loss_epoch)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]}, Validation Loss: {val_losses[-1]}, Train Accuracy: {train_accuracy}, Validation Accuracy: {val_accuracy}")
```

## Setup and Initialization

### Loss Function
- `loss_fn = nn.CrossEntropyLoss()`
  - The loss function used is CrossEntropyLoss, which combines LogSoftmax and NLLLoss in one single class, suitable for classification tasks with C classes.

### Optimizer
- `optimizer = optim.Adam(model.parameters(), lr=3e-5, weight_decay=1e-4)`
  - An Adam optimizer is used for adjusting the weights of the network, with a learning rate of `3e-5` and a weight decay of `1e-4`. This helps in controlling overfitting.

### Learning Rate Scheduler
- `scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, cooldown=1, min_lr=3e-8, eps=1e-09)`
  - This scheduler reduces the learning rate when a metric has stopped improving. The `mode` 'min' indicates that it will monitor a value that should decrease. The learning rate is reduced by a factor of `0.2` after a `patience` of 3 epochs if no improvement is seen. The cooldown is 1 epoch, and the minimum possible learning rate is `3e-8`.

### Training Parameters
- `num_epochs = 15`
  - Specifies the total number of training epochs.
- `best_val_accuracy = 0`
  - Tracks the highest validation accuracy achieved across epochs for model comparison.
- `epochs_no_improve = 0`
  - Counter for the number of consecutive epochs without improvement in validation accuracy.
- `early_stop_threshold = 3`
  - Specifies the number of consecutive epochs without improvement after which training will be stopped early.

## Training Loop

### Epochs Loop
- Iterates through each epoch of training, performing training and validation steps:

#### Training Phase
- Sets the model to training mode using `model.train()`.
- Initializes `train_loss`, `correct`, and `total` counters.
- Iterates over the training data loader:
  - Fetches inputs and labels.
  - Clears old gradients from the optimizer.
  - Performs a forward pass through the model.
  - Computes the loss.
  - Backpropagates the gradients.
  - Updates the model parameters.
  - Accumulates the total training loss and computes the accuracy.

#### Validation Phase
- Sets the model to evaluation mode using `model.eval()`.
- Disables gradient calculation to save memory and computations.
- Iterates over the validation data loader:
  - Performs forward passes with validation data.
  - Computes validation loss and accumulates results.
  - Computes validation accuracy.

### Early Stopping Check
- Checks if the current validation accuracy exceeds the best observed. If not, increments the no-improvement counter.
- If the counter reaches the early stop threshold, prints a message and exits the loop.

### Scheduler Step
- Updates the learning rate based on the average validation loss for the epoch.

### Epoch Summary
- Prints a summary of the epoch's results, including losses and accuracies for both training and validation.

This structured description of the training loop gives a clear insight into each step involved in training the neural network model, including the dynamic adjustments made to the learning rate and the conditions for early stopping based on validation performance.
***
2. [bird_sighting_prediction.ipynb](https://github.com/anthonyjarjar/Chloris/blob/main/api/src/jupyter%20notebooks/bird_sighting_prediction.ipynb)
    ```python
    oc_svm = SGDOneClassSVM(nu=0.01, max_iter=1000, tol=1e-4, learning_rate='optimal', shuffle=True, verbose=1, random_state=42)
    
    batch_size = 1000
    
    num_batches = int(np.ceil(len(X_scaled) / batch_size))
    
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(X_scaled))
    
        X_batch = X_train[start_idx:end_idx]
    
        if X_batch.shape[0] == 0:
            break
        oc_svm.partial_fit(X_batch)
    ```
    
## Overview
This script sets up and trains an SGD (Stochastic Gradient Descent) One-Class SVM on a dataset `X_scaled` using mini-batches. This approach is particularly suitable for large datasets that do not fit into memory all at once.

## Model Configuration
- `oc_svm = SGDOneClassSVM(nu=0.01, max_iter=1000, tol=1e-4, learning_rate='optimal', shuffle=True, verbose=1, random_state=42)`
  - An SGDOneClassSVM model is initialized with the following parameters:
    - `nu=0.01`: An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.
    - `max_iter=1000`: The maximum number of passes over the training data.
    - `tol=1e-4`: The stopping criterion for the model if training loss is less than this threshold.
    - `learning_rate='optimal'`: Specifies how the learning rate is set. 'Optimal' automatically adjusts the learning rate based on the data.
    - `shuffle=True`: Enables shuffling of the data before each epoch, which is beneficial for stochastic gradient methods.
    - `verbose=1`: Enables verbose output in the console, useful for monitoring training progress.
    - `random_state=42`: Sets a seed for reproducibility of shuffling and other random operations.

## Batch Processing Setup
- `batch_size = 1000`
  - Defines the number of samples per batch.
- `num_batches = int(np.ceil(len(X_scaled) / batch_size))`
  - Computes the total number of batches needed to process the entire dataset. This is calculated by dividing the total number of samples by the batch size and rounding up to ensure all samples are included.

## Training Loop
- `for i in range(num_batches):`
  - Iterates over each batch of the dataset.
  - `start_idx = i * batch_size`: Calculates the start index of the current batch.
  - `end_idx = min((i + 1) * batch_size, len(X_scaled))`: Calculates the end index of the current batch, ensuring it does not exceed the total number of samples.
  - `X_batch = X_train[start_idx:end_idx]`: Slices the training data to get the current batch.
  - Condition Check:
    - `if X_batch.shape[0] == 0:`
      - If the batch is empty (which can occur if the start index equals the dataset size), the loop breaks early.
  - `oc_svm.partial_fit(X_batch)`
    - Trains the model incrementally on the current batch. This method is useful for online learning or when the dataset is too large to fit into memory at once.
