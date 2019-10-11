#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "linear_regression.h"

/* Set Learning rate of linear regression. */
void set_learning_rate_lin(struct linreg_dataset_t *a, float b)
{
	// Set learning rate
	a->learning_rate = b;
}

/* Implementing feature scaling and normalization on a given dataset. */
void feature_scaling_and_normalization(struct linreg_dataset_t *a, int sampledatasize)
{
	// Calculating the mean value for each feature
	float sum1 = 0, sum2 = 0;
	for (int i = 0; i < sampledatasize; i++)
	{
		sum1 += a->feat1[i];
		sum2 += a->feat2[i];
	}
	float mean_val1 = sum1 / sampledatasize, mean_val2 = sum2 / sampledatasize;

	float std_sq1 = 0, std_sq2 = 0; // Stands for standard deviation squared
	for (int i = 0; i < sampledatasize; i++)
	{
		std_sq1 += pow(a->feat1[i] - mean_val1, 2);
		std_sq2 += pow(a->feat2[i] - mean_val2, 2);
	}

	// New value: the original subtracted by the mean value, splitted by the the std
	for (int i = 0; i < sampledatasize; i++)
	{
		a->feat1[i] = (a->feat1[i] - mean_val1) / sqrt(std_sq1);
		a->feat1[i] = (a->feat2[i] - mean_val2) / sqrt(std_sq2);
	}
}

/* Perform linear_regression */
void linear_regression(struct linreg_dataset_t *a, int sampledatasize)
{
	float loss_sum = 0;
	// Feature scaling and normalization of the dataset for improved preformence
	feature_scaling_and_normalization(a, sampledatasize);
	// Loop linear regression as many as sampledatasize
	for (int i = 0; i < sampledatasize; i++)
	{
		// Hypothesis function: hyp = B0 + B1*feat1 + B2*feat2
		float hypothesis = a->B0 + a->B1 * (a->feat1[i]) + a->B2 * (a->feat2[i]);
	
		// Calculate hypothesis error against actual output
		float err_lin = hypothesis - a->sys_output[i];  
		
		// Gradient descent: update each parameters to minimize error
		a->B0 = a->B0 - a->learning_rate * err_lin;
		a->B1 = a->B1 - a->learning_rate * err_lin * (a->feat1[i]);
		a->B2 = a->B2 - a->learning_rate * err_lin * (a->feat2[i]);
		
		// Calculate loss function of each sample and sum it up: (hyp - y)^2
		loss_sum += (hypothesis - (a->sys_output[i]))*(hypothesis - (a->sys_output[i]));
	}
	
	// Compute cost function
	a->cost_function = loss_sum/sampledatasize;
}



