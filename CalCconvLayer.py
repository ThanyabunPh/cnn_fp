import sys

conv_layer = 0
pooling = 0

class convnet:
    def __init__(self):
        self.width = 32
        self.height = 24

        self.input_channels = 1


    def conv(self, channels, kernel, stride, padding, size=None):

        in_out_channels = [int(i) for i in channels]

        global conv_layer
        conv_layer += 1

        self.kernel = [int(i) for i in kernel]
        self.width = size[0] if size else self.width
        self.height = size[1] if size else self.height
        
        self.stride = stride
        self.padding = padding


        output_width = (round(self.width - self.kernel[0] + (2 * self.padding)) / self.stride) + 1
        output_height = (round(self.height - self.kernel[1] + (2 * self.padding)) / self.stride) + 1

        print(f"Convolution Layer {conv_layer}\n")
        print(f"input_channels: {self.input_channels} | output_channels: {self.output_channels} | kernel: {self.kernel} | stride: {self.stride} | padding: {self.padding}\n")
        print(f"output_width: ({self.width} - {self.kernel[0]} + (2 x {self.padding})) / {self.stride} + 1 = {round(output_width)}")
        print(f"output_height: ({self.height} - {self.kernel[1]} + (2 x {self.padding})) / {self.stride} + 1 = {round(output_height)}\n")

        # conv_layer += 1
        return round(output_width), round(output_height)

    def maxpool(self, maxpool_stride, maxpool_kernel, size):
        output_width, output_height = size[0], size[1]

        self.maxpool_stride = maxpool_stride
        self.maxpool_kernel = [int(i) for i in maxpool_kernel]

        global pooling
        pooling += 1

        pool_width = (round(output_width - self.maxpool_kernel[0]) / self.maxpool_stride) + 1
        pool_height = (round(output_height - self.maxpool_kernel[1]) / self.maxpool_stride) + 1

        print(f"Pooling Layer {pooling}\n")
        print(f"pool_width: (({output_width} - {self.maxpool_kernel[0]}) / {self.maxpool_stride}) + 1 = {round(pool_width)}")
        print(f"pool_height: (({output_height} - {self.maxpool_kernel[1]}) / {self.maxpool_stride}) + 1 = {round(pool_height)}\n")

        return round(pool_width), round(pool_height), self.output_channels

    def flatten(self, size, Classified):
        self.size = size
        self.Classified = Classified
        print(f"\nFlattened size: {self.size[0]} x {self.size[1]} x {self.Classified} = {self.size[0] * self.size[1] * self.Classified}\n")
        return self.size[0] * self.size[1]



def step():
        
    print("#" * 50)
    conv1 = convnet().conv(channels=[1, 32], kernel=[4, 4], stride=1, padding=2)
    pooling1 = convnet().maxpool(maxpool_stride=2, maxpool_kernel=[3, 3], size=conv1)


    print("#" * 50)
    conv2 = convnet().conv(kernel=[3, 3], stride=1, padding=2, size=pooling1)
    pooling2 = convnet().maxpool(maxpool_stride=2, maxpool_kernel=[2, 2], size=conv2)

    print("#" * 50)
    conv3 = convnet().conv(kernel=[2, 2], stride=1, padding=1, size=pooling2)

    print("#" * 50)
    flat1 = convnet().flatten(size=conv3, Classified=4)




if __name__ == "__main__":
    step()



