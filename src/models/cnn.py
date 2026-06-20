import torch.nn as nn

def create_model(dropout_rate):

    model = nn.Sequential()
    
    ### First convolution layer
    model.add_module(
        name='conv1',
        module=nn.Conv2d(
            in_channels=3, out_channels=32,
            kernel_size=3, padding=1
        )
    )
    
    model.add_module(
        name='relu1',
        module=nn.ReLU()
    )
    
    model.add_module(
        name='pool1',
        module=nn.MaxPool2d(kernel_size=2)
    )
    
    model.add_module(
        name='dropout1',
        module=nn.Dropout(p=dropout_rate)
    )
    
    ### Second convolution layer
    model.add_module(
        name='conv2',
        module=nn.Conv2d(
            in_channels=32, out_channels=64,
            kernel_size=3, padding=1
        )
    )
    
    model.add_module(
        name='relu2',
        module=nn.ReLU()
    )
    
    model.add_module(
        name='pool2',
        module=nn.MaxPool2d(kernel_size=2)
    )
    
    model.add_module(
        name='dropout2',
        module=nn.Dropout(p=dropout_rate)
    )
    
    ### Third convolution layer
    model.add_module(
        name='conv3',
        module=nn.Conv2d(in_channels=64, out_channels=128,
                         kernel_size=3, padding=1)
    )
    
    model.add_module(
        name='relu3',
        module=nn.ReLU()
    )
    
    model.add_module(
        name='pool3',
        module=nn.MaxPool2d(kernel_size=2)
    )
    
    ### Fourth convolution
    model.add_module(
        name='conv4',
        module=nn.Conv2d(in_channels=128, out_channels=256,
                         kernel_size=3, padding=1)
    )
    
    model.add_module(
        name='relu4',
        module=nn.ReLU()
    )
    
    # in this situation - we'd need our fully-connected layer to take in 256*8*8 = 16384 neurons (after flattening)
    # we can simplify this by using global average pooling with kernel size the size of the feautre map
    #     computes the average of each map (only one output value)
    #     this will reduce our input to dense layer to 256 neurons
    model.add_module(
        name='pool4',
        module=nn.AvgPool2d(kernel_size=8)
    )
    
    # flatten 256x1x1 output size of pooling layer
    model.add_module(
        name='flatten',
        module=nn.Flatten()
    )
    
    # finally, add a dense layer to get single output
    model.add_module(
        name='fc',
        module=nn.Linear(in_features=256, out_features=1)
    )
    
    # add sigmoid as activation (output in range [0, 1])
    model.add_module(
        name='sigmoid',
        module=nn.Sigmoid()
    )
    
    return model