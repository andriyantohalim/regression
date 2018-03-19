#include <stdio.h>
#include <stdlib.h>
#include "linear_regression.h"
#include "logistic_regression.h"

// Compilation command:
// gcc main.c linear_regression.c logistic_regression.c -o main -lm && ./main


/* ==================== ONE DAY WORTH OF EXAMPLE DATA RECORDED FROM ADC ==================== */
// Clock day-time
float x1[] = { 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}; 
// Environment (outdoor) temperature sensor 
float x2[] = {25, 25, 24, 25, 26, 27, 27, 28, 29, 30, 31, 32, 33, 32, 33, 32, 31, 30, 30, 29, 29, 27, 27, 26};
// Occupancy sensor. ON = present, OFF = nobody in the room
float x3[] = { 0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0};

// Aircon temperature setting in degree C
float y0_aircon[] = {27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 26, 25, 25, 25, 25, 25, 25, 25, 26, 27, 27, 27, 27, 27};
// Aircon states (ON/OFF). ON = 1, OFF = 0
float y1_aircon[] = { 0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0};
/* ==================== ONE DAY WORTH OF EXAMPLE DATA RECORDED FROM ADC ==================== */

//system_dataset_t latest_readings;

int epoch = 150000;

int main(void)
{

	// Initialize linear regression dataset. Assign inputs/output here.
    linreg_dataset_t today_dataset = 
	{
		.feat1 = x1,				// Assign linreg feat1 to x1
		.feat2 = x2,				// Assign linreg feat1 to x2
		.sys_output = y0_aircon		// Assign linreg sysout to y0_aircon
	};

	// Set learning rate for linear regression
    set_learning_rate_lin(&today_dataset, 0.001);
    
	// Initialize logistic regression dataset. Assign inputs/output here.
	logreg_dataset_t today_OnOff = 
	{
		.feat1 = x1,				// Assign logreg feat1 to x1
		.feat2 = x2,				// Assign logreg feat2 to x2
		.feat3 = x3,				// Assign logreg feat3 to x3
		.sys_output = y1_aircon		// Assign logreg sysout to y1_aircon
	};
    
    // Set learning rate for logistic regression
	set_learning_rate_log(&today_OnOff, 0.001);
	
	for (int i = 0; i <= epoch; i++)
	{
		printf ("Epoch#%d\n", i);
			
		linear_regression(&today_dataset, 24);	
		logistic_regression(&today_OnOff, 24);	
			
		printf ("LINREG\tB0 = %2.3f,\tB1 = %2.3f,\tB2 = %2.3f,             \tCost = %2.3f\n", today_dataset.B0, today_dataset.B1, today_dataset.B2, today_dataset.cost_function);
		printf ("LOGREG\tB0 = %2.3f,\tB1 = %2.3f,\tB2 = %2.3f,\tB3 = %2.3f,\tCost = %2.3f\n", today_OnOff.B0, today_OnOff.B1, today_OnOff.B2, today_OnOff.B3, today_OnOff.cost_function);
		printf ("\n");	
	}
	
	
	
	while(1)
	{

			
	}
	
  
    return 0;
}
