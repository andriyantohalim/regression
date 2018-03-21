#ifndef LINEAR_REGRESSION_H_  
#define LINEAR_REGRESSION_H_


/* DATA STRUCTURE FROM THE SYSTEM */
typedef struct system_dataset_t 
{
	int sys_day;
	int sys_hour;
	int sys_minute;
	int sys_seconds;

	int temperature_ambient;	// measured temperature by the system
	int temperature_setting;	// temperature value set by user
	int occupancy_sensor;
	
	int sys_status;
	int sys_user_input_switch;
} system_dataset_t;

/* DATA STRUCTURE FOR LINEAR REGRESSION */
typedef struct linreg_dataset_t
{
	float *feat1;	// pointer to be assigned to MCU input signals
	float *feat2; 	// pointer to be assigned to MCU input signals
	
	float *sys_output;
	
	float B0;	// linreg weightage
	float B1;	// linreg weightage
	float B2;	// linreg weightage
	
	float learning_rate;
	
	float cost_function;
} linreg_dataset_t;



void set_learning_rate_lin(struct linreg_dataset_t *a, float b);
void linear_regression(struct linreg_dataset_t *a, int sampledatasize);

/* TODO: Functions to be implemented on the MCU/MPU later */ 
void get_latest_systemdata();
void save_linear_regression_states();
void load_linear_regression_states();
void perform_inference();
/* TODO: Functions to be implemented on the MCU/MPU later */ 


#endif // LINEAR_REGRESSION
