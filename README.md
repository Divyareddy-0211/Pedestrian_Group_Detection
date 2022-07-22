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

#### Here Loading the groups.txt file which has our ground truth values.(original groups)
```
with open('C:\\Users\\sagar\\Desktop\\Project_Final_Group_4\\groups.txt') as f:
    lines = f.readlines()
lines = [x.strip() for x in lines]
lines = [x for x in lines if x]
ground_truth = []
for i in range(len(lines)):
    ground_truth.append(lines[i].split(' '))
    # convert to integer
    ground_truth[i] = [int(x) for x in ground_truth[i]]
```

#### Taking the unique ids of the frames and agents for further analysis in the code
```
frame_id = data['frame_id'].unique()
agent_ids = sorted(data['agent_id'].unique())
```

#### Here dropping the unuseful columns for our analysis
```
data = data.drop(columns=['label','scene_id','frame_id','vel_x', 'vel_y'])
```

#### pivoting the data here using timestamp as a index
###### Here we are using timestamp as the index and columns as the agent_id and dividing the dataframe between pos_x and pos_y values for our further analysis
###### and also putting the extreme values in the data where it is a null value and converting the whole data into an array
```
data = data.pivot(index = 'timestamp',columns='agent_id',values = ['pos_x','pos_y'])
data = data.fillna(50) #etreme value
X = np.array(data)
```


#### Collecting all points in one array
###### Now y is in shape of two columns it has the pos_x and pos_y values of the agents which are drawn from the numpy array X
```
Y = []
for i in range(X.shape[0]):
    for j in range(X.shape[1]//2):
        Y+=[[X[i,j],X[i,j+X.shape[1]//2]]]
Y = np.array(Y)
Y
```


####
```

```


####
```

```

####
```

```
