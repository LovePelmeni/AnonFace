import numpy 

class ReceptiveFieldCalculator(object):

    def __init__(self, total_layers: list):
        self.strides = numpy.zeros(total_layers)
        self.rfs = numpy.ones(total_layers)
        self.last_stride_product = 1

    def calculate_rfs(self, input_layers: list):
        """
        Calculates Receptive Field for each individual
        layer in the network.

        Limitations:
            - does not include dilation rate, in case
            dilation convolutions are leveraged.
        
        Pros:
            - factors pooling and standard convolutional layers
        """
        for idx, layer in enumerate(input_layers):
            self.strides[idx] = layer.stride
            self.rfs[idx] = self.rfs[idx-1] + (layer.kernel_size - 1) * self.last_stride_product
            self.last_stride_product *= layer.stride
        return self.rfs