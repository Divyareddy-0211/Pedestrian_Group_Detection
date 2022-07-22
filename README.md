# Pedestrian_Group_Detection

## Table of contents
* [Project Overview](#project-overview)
* [Project Execution](#project-execution)
* [Code Explanation](#code-explanation)

## Project Overview

## Project Execution
Just check the paths of the CSV File and Groups.txt file in the code.
Direct them to the respective files.

We can execute the code in Any IDE directly. Just run the file.
At the end we can see all the plots generated at each eps value their different "r" threshold values 
and their respective IOU Values.

"or"

we can also execute the file in cmd.

"path to the folder">python Project_final.py.  Executing this will excute the complete file.
Executing the code in cmd generates the plots in sequential manner. You need to close one plot to see 
the next plot. In total there are 30 plots for 30 different eps_values.


Total execution run time for the code is approximaytely 6 minutes. 
At different eps_values ranging from 0.1 to 3.0

## Code Explanation
#### Importing all the necessary libraries 
```
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
```
#### Loading our data using pandas libraray.
```
data = pd.read_csv(r'C:\Users\sagar\Desktop\Project_Final_Group_4\test_eth_2.5fls.csv' )
```


