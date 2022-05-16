#include "dataset.h"

#define XNOR_GATE

Dataset::Dataset(size_t batch_size, size_t number_of_batches) : 
	batch_size(batch_size), number_of_batches(number_of_batches)
{
	for(int i = 0; i < number_of_batches; i++)
	{
		batches.push_back(Matrix(Shape(3, batch_size * 8)));
		targets.push_back(Matrix(Shape(1, batch_size * 8)));

		batches[i].allocateMemory();
		targets[i].allocateMemory();

		float X1, X2, S1, Y;

		for (int k = 0; k < batch_size * 8; k++)
		{

			switch (k%8)
			{
				case 0:
					X1 = 0;
					X2 = 0;
					S1 = 0;
					Y  = 0;
					break;
				case 1:
					X1 = 0;
					X2 = 0;
					S1 = 1;
					Y  = 0;
					break;
				case 2:
					X1 = 0;
					X2 = 1;
					S1 = 0;
					Y  = 0;
					break;
				case 3:
					X1 = 0;
					X2 = 1;
					S1 = 1;
					Y  = 1;
					break;
				case 4:
					X1 = 1;
					X2 = 0;
					S1 = 0;
					Y  = 1;
					break;
				case 5:
					X1 = 1;
					X2 = 0;
					S1 = 1;
					Y  = 0;
					break;
				case 6:
					X1 = 1;
					X2 = 1;
					S1 = 0;
					Y  = 1;
					break;
				case 7:
					X1 = 1;
					X2 = 1;
					S1 = 1;
					Y  = 1;
					break;
				default:
					X1 = 0;
					X2 = 0;
					S1 = 0;
					Y  = 0;
					break;
			}


			batches[i].data_host.at<float>(0, k) = X1;
			batches[i].data_host.at<float>(1, k) = X2;
			batches[i].data_host.at<float>(2, k) = S1;

			targets[i].data_host.at<float>(0, k) = Y;
		}
		
		batches[i].copyHostToDevice();
		targets[i].copyHostToDevice();
	}
}

int Dataset::getNumOfBatches() 
{
	return number_of_batches;
}

std::vector<Matrix>& Dataset::getBatches() 
{
	return batches;
}

std::vector<Matrix>& Dataset::getTargets() 
{
	return targets;
}
