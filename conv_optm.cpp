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


void conv_optm(hls::stream<axis_data> &in_A, hls::stream<axis_data> &out_C) {

	#pragma HLS INTERFACE ap_ctrl_none port=return
	#pragma HLS INTERFACE axis register both port=in_A
	#pragma HLS INTERFACE axis register both port=out_C

	// matrices to store inputs and outputs
	 int row;
	 int col;
	 int x = (((h+(2*p))-kernel)/s)+1;
	 int y = (((w+(2*p))-kernel)/s)+1;
	 axis_data str;


	 Mat_Dtype input_image[h][w];
	 #pragma HLS ARRAY_PARTITION variable=input_image complete dim=2
	 Mat_Dtype filter[kernel][kernel];
	 #pragma HLS ARRAY_PARTITION variable=filter complete dim=1
	 Mat_Dtype output[x][y];  // Matrix to store the output

	// read data for Input Image

	loop_input_A1: for(row=0; row < h; row++){
		loop_input_A2: for(col=0; col < w; col++){
			#pragma HLS PIPELINE
			str = in_A.read();
			input_image[row][col] = str.data;
		}
	}

	// read data for Filter Matrix

	loop_input_B1: for(row=0; row < kernel; row++){
		loop_input_B2: for(col=0; col < kernel; col++){
			#pragma HLS PIPELINE
			str = in_A.read();
			filter[row][col] = str.data;
		}
	}

	// Perform convolution
	loop1: for (int row = 0; row < x; ++row) {
		loop2: for (int col = 0; col < y; ++col) {
			#pragma HLS PIPELINE II=2
	        	output[row][col] = 0;
	            	loop3: for (int i = 0; i < kernel; ++i) {
	                	loop4: for (int j = 0; j < kernel; ++j) {
					#pragma HLS PIPELINE
	                		output[row][col]+= input_image[row + i][col + j] * filter[i][j];
	                	}
	            	}
	        }
	}
	// Write output matrix to AXI stream
	for (int row = 0; row < x; row++) {
		for (int col = 0; col < y; col++) {
			#pragma HLS PIPELINE
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
