#include <hls_stream.h>
#include <ap_int.h>
#include <stdio.h>


#define kernel 3
#define w 5 // Input image matrix width
#define h 5 // Input image matrix height
#define p 0 // Padding applied to input image matrix
#define s 1 // Stride for filter

typedef double Mat_Dtype;

struct axis_data{
	Mat_Dtype data;
	ap_uint<1> last;

};


void conv(hls::stream<axis_data> &in_A,  hls::stream<axis_data> &out_C) {

    #pragma HLS INTERFACE ap_ctrl_none port=return
    #pragma HLS INTERFACE axis register both port=in_A
    #pragma HLS INTERFACE axis register both port=out_C

    int rows;
    int cols;
    int x = (((h+(2*p))-kernel)/s)+1;
    int y = (((w+(2*p))-kernel)/s)+1;
    axis_data str;


    Mat_Dtype input_image[h][w];
	#pragma HLS array_partition variable=input_image complete dim=2
    Mat_Dtype filter[kernel][kernel];
	#pragma HLS array_partition variable=filter complete dim=1
    Mat_Dtype output[x][y];  // Matrix to store the output

    // Read input matrix from AXI stream
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
        	str = in_A.read();
            input_image[i][j] = str.data;
        }
    }

    // Read kernel matrix from AXI stream
    for (int i = 0; i < kernel; i++) {
        for (int j = 0; j < kernel; j++) {
            str = in_A.read();
            filter[i][j] =str.data;
        }

    }


    // Perform convolution
    for (int row = 0; row < x; ++row) {
        for (int col = 0; col < y; ++col) {
        	output[row][col] = 0;
            for (int i = 0; i < kernel; ++i) {
                for (int j = 0; j < kernel; ++j) {
					#pragma HLS PIPELINE
                	output[row][col]+= input_image[row + i][col + j] * filter[i][j];
                }
            }
        }
    }

    // Write output matrix to AXI stream
    for (int row = 0; row < x; row++) {
        for (int col = 0; col < y; col++) {
            str.data = output[row][col];
            if((row == x-1)&&(col == y-1)){
            	str.last = 1;
            }

            else{
            	str.last = 0;
            }
            out_C.write(str);



        }
    }
}
