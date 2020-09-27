# CISC684_Project1
## Authors:
- Evan MacBride
- Collin Meese
- Matt Leinhauser

Contact: {macbride, cmeese, mattl} @udel.edu

## How to Run:
There are two ways to run this code:
1. Run with the automated bash script on the terminal when you are within the project directory.
    
    a. To make sure you can execute the script use the command:
    
    `chmod +x run_experiments.sh`
    
    b. Execute the script on the terminal with the following command:
    `./run_experiments.sh`
     
2. Run on the command line using the following: 

`python3  project1_executable.py L K <train_set_path> <test_set_path> <validation_set_path> [yes, no]`

-  An example run command is the following:
    
    `python3 project1_executable.py 3 4 data_sets1/data_sets1/training_set.csv data_sets1/data_sets1/test_set.csv data_sets1/data_sets1/validation_set.csv yes`

where:
- L = 3
- K = 4
- train_set = data_sets1/data_sets1/training_set.csv
- test_set = data_sets1/data_sets1/test_set.csv
- validation_set = data_sets1/data_sets1/validation_set.csv
- to_print = yes
