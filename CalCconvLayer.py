import sys

conv_layer = 0
pooling = 0

class convnet:
    def __init__(self):
        self.width = 28
        self.height = 28
        self.conv_layer = 0

    def conv(self, kernel, stride, padding, size=None):

        # global conv_layer
        self.conv_layer += 1

        self.kernel = [int(i) for i in kernel]
        self.width = size[0] if size else self.width
        self.height = size[1] if size else self.height
        
        self.stride = stride
        self.padding = padding

        # 
        output_width = (round(self.width - self.kernel[0] + (2 * self.padding)) / self.stride) + 1
        output_height = (round(self.height - self.kernel[1] + (2 * self.padding)) / self.stride) + 1

        print(f"Convolution Layer {self.conv_layer}\n")
        # print(f"input_channels: {self.input_channels} | output_channels: {self.output_channels} | kernel: {self.kernel} | stride: {self.stride} | padding: {self.padding}\n")
        print(f"output_width: ({self.width} - {self.kernel[0]} + (2 x {self.padding})) / {self.stride} + 1 = {round(output_width)}")
        print(f"output_height: ({self.height} - {self.kernel[1]} + (2 x {self.padding})) / {self.stride} + 1 = {round(output_height)}\n")

        return round(output_width), round(output_height)

    def pooling(self, pooling_kernel, size, pooling_stride=0):
        output_width, output_height = size[0], size[1]
        self.pooling_stride = pooling_stride
        self.pooling_kernel = [int(i) for i in pooling_kernel]

        
        global pooling
        pooling += 1

        if self.pooling_stride != 0:
            print("Pooling stride is not 0")
            pool_width = (round(output_width - self.pooling_kernel[0]) / self.pooling_stride) + 1
            pool_height = (round(output_height - self.pooling_kernel[1]) / self.pooling_stride) + 1
    
            print(f"Pooling Layer {pooling}\n")
            print(f"pool_width: (({output_width} - {self.pooling_kernel[0]}) / {self.pooling_stride}) + 1 = {round(pool_width)}")
            print(f"pool_height: (({output_height} - {self.pooling_kernel[1]}) / {self.pooling_stride}) + 1 = {round(pool_height)}\n")
    
            return round(pool_width), round(pool_height)

        else:
            print("Pooling stride is 0")
            pool_width = round(output_width / self.pooling_kernel[0])
            pool_height = round(output_height / self.pooling_kernel[1])
            print(f"Pooling Layer {pooling}\n")
            print(f"pool_width: {output_width} / {self.pooling_kernel[0]} = {round(pool_width)}")
            print(f"pool_height: {output_height} / {self.pooling_kernel[1]} = {round(pool_height)}\n")
            return round(pool_width), round(pool_height)

            

    def flatten(self, size, Classified):
        self.size = size
        self.Classified = Classified
        print(f"\nFlattened size: {self.size[0]} x {self.size[1]} x {self.Classified} = {self.size[0] * self.size[1] * self.Classified}\n")
        return self.size[0] * self.size[1]



def step():
        
    print("#" * 50)
    conv1 = convnet().conv(kernel=[5,5], stride=1, padding=0)
    pooling1 = convnet().pooling(pooling_kernel=[2, 2], size=conv1, pooling_stride=1)

    conv2 = convnet().conv(kernel=[5,5], stride=1, padding=0, size=pooling1)
    pooling2 = convnet().pooling(pooling_kernel=[2, 2], size=conv2, pooling_stride=1)




if __name__ == "__main__":
    step()



