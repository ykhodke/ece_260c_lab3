#include <iostream>
#include <random>
#include <assert.h>

#include "linear_layer.h"
#include "nn_exception.h"

using namespace std;
using namespace cv;

__global__ void linearLayerForward(float *W, float* input, float* output, float* b,
									const int W_rows, const int W_cols,
									const int input_rows, const int input_cols) 
{
    //TODO: complete the linear layer forward propagation
		int o_r = blockIdx.y * blockDim.y + threadIdx.y;
		int o_c = blockIdx.x * blockDim.x + threadIdx.x;

		//output dimensions > [weight rows, input columns]
		int or_max = W_rows;
		int oc_max = input_cols;

		float temp_out = 0.0f;

		if(o_r < or_max && o_c < oc_max)
		{
			for (int k = 0; k < W_cols; k++)
			{
				temp_out += W[o_r * W_cols + k] * input[k * input_cols + o_c];
			}

			output[o_r*oc_max + o_c] = temp_out + b[o_r];
		}
}

__global__ void linearLayerBackprop(float *W, float* eB, float* eA,
									const int W_rows, const int W_cols,
									const int eB_rows, const int eB_cols) 
{
    //TODO: complete the linear layer backpropagation
		int o_r = blockIdx.y * blockDim.y + threadIdx.y;
		int o_c = blockIdx.x * blockDim.x + threadIdx.x;

		int ea_rows = W_cols;
		int ea_cols = eB_cols;
		
		float ea_value = 0.0f;

		if (o_r < ea_rows && o_c < ea_cols)
		{
			for (int k = 0; k < W_rows; k++)
			{
				ea_value += W[k * W_cols + o_r] * eB[k * eB_cols + o_c];
			}
			eA[o_r * ea_cols + o_c] = ea_value;
		}
}

__global__ void linearLayerUpdateWeights(float *eB, float* input, float* W,
									const int eB_rows, const int eB_cols,
									const int input_rows, const int input_cols, float learning_rate)
{
    //TODO: complete the gradient descent for weight updates
		int o_r = blockIdx.y * blockDim.y + threadIdx.y;
		int o_c = blockIdx.x * blockDim.x + threadIdx.x;

		float dw = 0.0f;
		
		int w_r = eB_rows;
		int w_c = input_rows;

		if (o_r < w_r && o_c < w_c)
		{
			for (int k = 0; k < eB_cols; k++)
			{
				dw += eB [o_r * eB_cols + k] * input [ o_c * input_cols + k]; 
			}
			W[o_r*w_c + o_c] -= learning_rate* (dw/input_rows);
		}
}

__global__ void linearLayerUpdateBias(float *eB, float* b,
									const int eB_rows, const int eB_cols,
									const int b_rows, float learning_rate)
{
    //TODO: complete the gradient descent for bias updates
		 int index = blockIdx.x * blockDim.x + threadIdx.x;

		 if (index < eB_rows*eB_cols)
		 {
			 int col = index % eB_cols;
			 int row = index / eB_cols;
			 atomicAdd(&b[row], -learning_rate * (eB[row * eB_cols + col] / eB_cols));
		 }
}

LinearLayer::LinearLayer(string name, Shape W_shape) 
{
	W_shape.transpose();
	
	Matrix weights(W_shape);
	Matrix bias(W_shape.rows, 1);

	this->W = weights;
	this->b = bias;

	this->name = name;
	b.allocateMemory();
	W.allocateMemory();
	initializeBiasWithZeros();
	initializeWeightsRandomly();
}

LinearLayer::~LinearLayer() {}

void LinearLayer::initializeWeightsRandomly() 
{
	
	float mean = 0.0;
	float stddev = 1.0;

	theRNG().state = time(NULL);
	randn(W.data_host, Scalar(mean), Scalar(stddev));

	W.copyHostToDevice();
}

void LinearLayer::initializeWeightsHalf() 
{
	W.data_host = Scalar(0.5f);

	W.copyHostToDevice();
}


void LinearLayer::initializeBiasWithZeros()
{
	
	b.data_host = Scalar(0.0f);

	b.copyHostToDevice();
}

Matrix& LinearLayer::forward(Matrix& input)
{
	assert(W.shape.cols == input.shape.rows);
	
	this->input = input;

	Shape output_shape(W.shape.rows, input.shape.cols);

	output.allocateMemoryIfNotAllocated(output_shape);

	computeAndStoreLayerOutput(input);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform linear layer forward propagation");

	return output;
}

void LinearLayer::computeAndStoreLayerOutput(Matrix& input) {
	dim3 block_size(8, 8);
	dim3 num_of_blocks(	(output.shape.cols + block_size.x - 1) / block_size.x,
						(output.shape.rows + block_size.y - 1) / block_size.y);

	linearLayerForward<<<num_of_blocks, block_size>>>( W.data_device,
															input.data_device,
															output.data_device,
															b.data_device,
															W.shape.rows, W.shape.cols,
															input.shape.rows, input.shape.cols);
}

Matrix& LinearLayer::backprop(Matrix& eB, float learning_rate)
{
	eA.allocateMemoryIfNotAllocated(input.shape);

	computeAndStoreBackpropError(eB);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform back propagation.");

	updateBias(eB, learning_rate);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform bias update.");

	updateWeights(eB, learning_rate);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform weights update.");

	return eA;
}


void LinearLayer::computeAndStoreBackpropError(Matrix& eB) {
	dim3 block_size(8, 8);
	dim3 num_of_blocks(	(input.shape.cols + block_size.x - 1) / block_size.x,
						(input.shape.rows + block_size.y - 1) / block_size.y);

	linearLayerBackprop<<<num_of_blocks, block_size>>>( W.data_device,
															eB.data_device,
															eA.data_device,
															W.shape.rows, W.shape.cols,
															eB.shape.rows, eB.shape.cols);
}

void LinearLayer::updateWeights(Matrix& eB, float learning_rate) {
	dim3 block_size(8, 8);
	dim3 num_of_blocks(	(W.shape.cols + block_size.x - 1) / block_size.x,
						(W.shape.rows + block_size.y - 1) / block_size.y);

	linearLayerUpdateWeights<<<num_of_blocks, block_size>>>(eB.data_device,
															input.data_device,
															W.data_device,
															eB.shape.rows, eB.shape.cols,
															input.shape.rows, input.shape.cols,
															learning_rate);
}

void LinearLayer::updateBias(Matrix& eB, float learning_rate) {
	dim3 block_size(256);
	dim3 num_of_blocks( (eB.shape.rows * eB.shape.cols + block_size.x - 1) / block_size.x);

	linearLayerUpdateBias<<<num_of_blocks, block_size>>>(eB.data_device,
															b.data_device,
															eB.shape.rows, eB.shape.cols,
															b.shape.rows, learning_rate);
}

int LinearLayer::getXDim() const {
	return W.shape.cols;
}

int LinearLayer::getYDim() const {
	return W.shape.rows;
}

Matrix LinearLayer::getWeightsMatrix() const {
	return W;
}

Matrix LinearLayer::getBiasVector() const {
	return b;
}
