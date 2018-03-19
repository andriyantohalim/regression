#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "logistic_regression.h"

/* Set Learning rate of logistic regression. */
void set_learning_rate_log(struct logreg_dataset_t *a, float b)
{
	// Set learning rate
	a->learning_rate = b;
}

/* Perform logistic_regression */
void logistic_regression(struct logreg_dataset_t *a, int sampledatasize)
{
	// Loop linear regression as many as sampledatasize
	for (int i = 0; i < sampledatasize; i++)
	{
		// Hypothesis (logistic) function: hyp = 1/(1 + e^(-(B0 + B1*feat1 + B2*feat2 + B3*feat3)))
		float linear_eqn = a->B0 + a->B1 * (a->feat1[i]) + a->B2 * (a->feat2[i]) + a->B3 * (a->feat3[i]) ;
		float denom = 1 + exp(-(linear_eqn));
		float hypothesis = 1/denom;
			
		// Calculate hypothesis error against actual output
		float err_log = hypothesis - a->sys_output[i];  
		
		// Gradient descent: update each parameters to minimize error
		a->B0 = a->B0 - a->learning_rate * err_log;
		a->B1 = a->B1 - a->learning_rate * err_log * (a->feat1[i]);
		a->B2 = a->B2 - a->learning_rate * err_log * (a->feat2[i]);
		a->B3 = a->B3 - a->learning_rate * err_log * (a->feat3[i]);
	
		// Calculate cost function in moving average mode (not as summation): -y*log(hyp) - (1-y)*log(1-hyp)
		a->cost_function = a->cost_function_prev - (a->sys_output[i] * log(hypothesis) + (1 - a->sys_output[i]) * log(1 - hypothesis));
		a->cost_function_prev = a->cost_function/2;  
	}
}



