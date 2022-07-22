# Pedestrian_Group_Detection

## Table of contents
* [Project Overview](#project-overview)
* [Project Execution](#project-execution)
* [Code Explanation](#code-explanation)
* [Best Parameter](#best-parameter)
* [IOU Plot](#Iou-plot)

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


#### Here we are applying our DBScan model
###### Here we are sending the data to our dbscan algorithm frame wise DBScan algorithm internally uses the Kneighbours algorithm with euclidean distances to 
###### group the agents with defined euclidean distance. 'eps' is the DBScan algorithm parameter which defines the euclidean distance.In the end we are adding all of ###### our results in the 'clusters' array coverting the clusters array to dataframe with index as the frame_id and the columns are the agents_id's.
```
 clusters = np.array([])
    a = X.shape[1]//2
    for ik in range(X.shape[0]):
        clustering = DBSCAN(eps = i, min_samples = 2).fit(Y[a*ik:a*(ik+1),:])
        clusters = np.append(clusters,np.array(clustering.labels_))
    clusters = clusters.reshape(X.shape[0],a)
    clusters
    
    cluster_df = pd.DataFrame(clusters,columns =agent_ids,index=frame_id)
    
    cluster_df["max_groups"] = cluster_df.max(axis=1)
```


#### Here we are grouping all the groups given by the DBScan algorithm
###### Here we are creating a group dict to collect the information of the people who are grouped together by the DBScan algorithm based on their Euclidean distances
###### and we are also taking the count how many times they are grouped together. This also means that they are coexisting in that frames_id.
```
    groups_dict = {}
    #indvidual_groups = {}
    for index, row in cluster_df.iterrows():
        total_groups = int(row["max_groups"])
        if total_groups > 0:
           # groups = []
            for group_id in range(1, total_groups + 1):
                group_set = list()
                for agent_id in agent_ids:
                    if row[agent_id] == group_id:
                        group_set.append(int(agent_id))
                if group_set:
                    key = tuple(group_set)
                 #   groups.append(group_set)
                    groups_dict[key] = (1 if key not in groups_dict else groups_dict[key] + 1)


```

#### Here we are caculating the coexisting time ratio for with a threshold we are defining the groups as a predicted groups. all the groups that are les than the threshold value are not considered.
###### here we are checking for all certain threshold values from 0.1 t 1.0 a based on these threshold values we are defining the groups given by the DBScan algorithm  as  predicted groups. Our Groups dict has all the groups and number of times they are considered as a group. Now this value is going to be our Numerator. and the denominator is the union of the distinct steps these agents in that group has taken. (Referred fom the Research Paper provided).
```
    rvalues =  np.arange(0.1, 1.01, 0.01)
    for rf in rvalues:
        print("threshold_value",rf)
        predicted_groups=[]
        complete_list = {}
        for key,value in groups_dict.items():
            distinct_steps = []
            groups = []
            for ke,val in groups_dict.items():
                groups.append((ke,val))
            for ij in key:
                for j in groups:
                   if ij in j[0]:
                      distinct_steps.append(j[1])
                      idx = groups.index(j)
                      groups.pop(idx)
            #calculating the r value
            indvidual_steps = sum(distinct_steps)
            complete_list[key] = (value,indvidual_steps)
            r = (value/indvidual_steps)
            if r >= rf:
                predicted_groups.append(list(key))
```

#### Here we are calculating our IOU(Intersection over Union) Metric
```
intersection = []
        matched_values = []
        iou=0
        for p in predicted_groups:
            for g in ground_truth:
                if set(p) == set(g):
                    intersection.append(p)
                    matched_values.append(g)
        remaning_values_ground_truth = len(ground_truth)-len(matched_values)
        remaining_values_predicted_groups = len(predicted_groups)-len(intersection)
        union_values = len(matched_values)+remaning_values_ground_truth+remaining_values_predicted_groups
        iou = len(intersection)/union_values            
        iou_values.append(iou)
        print("calculated_iou",iou)
        allv = (i,rf,iou)
        allvalues.append(allv)
        if str(i) not in matched_dict:
           matched_dict[str(i)] = [[[rf],[iou],matched_values,predicted_groups]]
        else:
            matched_dict[str(i)].append([[rf],[iou],matched_values,predicted_groups])
```
#### Here we are checking the maximum IOU value that we got for our different r values and the different eps values.
```
for i in allvalues:
    if i[2]==max(iou_values):
        print("iou is maximum at the eps_value = "+str(i[0])+" and corelation_value = "+str(i[1])+" with value",i[2]
```

#### Here we are plotting the graph for one eps value and different threshold values and their IOU values. We get multiple graphs at multiple eps values
```
for e in eps_values:
    #print(e)
    rvalues = []
    iouvalues = []
    for a in allvalues:
       # print(a)
        if e == a[0]:
            rvalues.append(a[1])
            iouvalues.append(a[2])
    ious = np.array(iouvalues)
    rval = np.array(rvalues)
    plt.plot(rval,ious)
    plt.xlabel('r Values')
    plt.ylabel('iou')
    plt.title("iou for different r values at epsvalue = "+str(e))
    plt.show()

```

## Best Parameter
###### iou is maximum at the eps_value = 1.5700000000000007 and corelation_value = 0.5099999999999998 with value 0.5753424657534246

## IOU Plot
![IOU](https://user-images.githubusercontent.com/84289491/180461997-17605e32-a106-4f3e-990e-4b2516c5a21b.png)



