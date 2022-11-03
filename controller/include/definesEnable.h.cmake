#ifndef ENABLE_DEFINES
#define ENABLE_DEFINES
#cmakedefine MIGRATION 	                      //Disable/Enable migration
#cmakedefine ERROR_CHECKING 	              //Disable/Enable Error Checking
#cmakedefine FREE_THREAD 	                  //Disable/Enable a thread that performs ref_dec
#cmakedefine ELASTICITY 	                  //Disable/Enable breakdown timers inside controller
#cmakedefine BREAKDOWNS_CONTROLLER 	          //Disable/Enable breakdown timers inside controller
#cmakedefine DATA_TRANSFER	                  //Disable/Enable prints for input-output data size
#cmakedefine REDUCTION_UNITS_FROM_EXEC_TIME	  //Reduction units according to execution time or statically defined as -1
#cmakedefine POWER_TIMER	 		 //Returns timers to be used for POWER-UTILZATION
#define BUILTINS_PATH "@BUILTINS_PATH@"	 		 //Path containing the buitin libs
#endif
