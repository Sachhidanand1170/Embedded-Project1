from pynq import Overlay
from pynq import allocate
import numpy as np
import math
import time


# ### Program the FPGA with the bit file

ol = Overlay("/home/xilinx/jupyter_notebooks/design_conv.bit")


# ### Check the IPs in the overaly (configuration provided by the bit file)

ol.ip_dict


# ### Create an instance of the DMA and define functions for sending and receiving the data

dma = ol.axi_dma_0

dma_send = ol.axi_dma_0.sendchannel
dma_recv = ol.axi_dma_0.recvchannel


# ### Define three arrays in the PS memory to store the data transfer
data_size_1 = 25
input_buffer1 = allocate(shape=(data_size_1,), dtype=float)
data_size_2 = 9
input_buffer2 = allocate(shape=(data_size_2,), dtype=float)
data_size_3 = 9
output_buffer = allocate(shape=(data_size_3,), dtype=float)

# ### Generate some test data

for i in range(data_size_1):
    input_buffer1[i] = i + 1
for i in range(data_size_2):
    input_buffer2[i] = i + 1


# ### Send the data

start = time.time()

dma_send.transfer(input_buffer1)
dma_send.idle

dma_send.transfer(input_buffer2)
dma_send.idle


# ### Receive the data from the Streaming FIFO
dma_recv.transfer(output_buffer)
dma.recvchannel.wait()

end = time.time()
fpga_run_time = end - start

A_matrix = input_buffer1.reshape((int(math.sqrt(data_size_1)),int(math.sqrt(data_size_1))))
B_matrix = input_buffer2.reshape((int(math.sqrt(data_size_2)),int(math.sqrt(data_size_2))))
Output_matrix = output_buffer.reshape((int(math.sqrt(data_size_3)),int(math.sqrt(data_size))))

print('A Matrix :')
print(A_matrix)
print("\n")

print('B matrix :')
print(B_matrix)
print("\n")


print('Out matrix :')
print(Output_matrix)
print("\n")


def perform_convolution(input_data, kernel):
    input_height, input_width = input_data.shape
    kernel_height, kernel_width = kernel.shape
    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1

    result = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            result[i, j] = np.sum(input_data[i:i+kernel_height, j:j+kernel_width] * kernel)

    return result


# Reshape input for convolution
input_matrix = input_buffer1.reshape((int(math.sqrt(data_size_1)), int(math.sqrt(data_size_1))))
kernel_matrix = input_buffer2.reshape((int(math.sqrt(data_size_2)), int(math.sqrt(data_size_2))))

# Perform convolution
result_convolution = perform_convolution(input_matrix, kernel_matrix)
print('Python convolution result :')
print(result_convolution)
print("\n")

# Check if the matrices are equal
are_matrices_equal = np.array_equal(Output_matrix, result_convolution)

# Print the result
if are_matrices_equal:
    print("The matrices are equal.")
else:
    print("The matrices are not equal.")

dma_recv.idle
print('FPGA run time: ', fpga_run_time)
print('ARM PS run time: ', ps_run_time)
