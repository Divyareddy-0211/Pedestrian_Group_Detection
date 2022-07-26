# importing required libraries
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
#initial data
data = pd.read_csv(r'C:\Users\sagar\Desktop\Project_Final_Group_4\test_eth_2.5fls.csv' )

# here all ground truth values
with open('C:\\Users\\sagar\\Desktop\\Project_Final_Group_4\\groups.txt') as f:
    lines = f.readlines()
lines = [x.strip() for x in lines]
lines = [x for x in lines if x]
ground_truth = []
for i in range(len(lines)):
    ground_truth.append(lines[i].split(' '))
    # convert to integer
    ground_truth[i] = [int(x) for x in ground_truth[i]]
    

frame_id = data['frame_id'].unique()
agent_ids = sorted(data['agent_id'].unique())

# drop unuseful columns 
data = data.drop(columns=['label','scene_id','frame_id','vel_x', 'vel_y'])


data = data.pivot(index = 'timestamp',columns='agent_id',values = ['pos_x','pos_y'])
data = data.fillna(50) #etreme value
X = np.array(data) # change to numpy array

# collecting all points in one array
Y = []
for i in range(X.shape[0]):
    for j in range(X.shape[1]//2):
        Y+=[[X[i,j],X[i,j+X.shape[1]//2]]]
Y = np.array(Y)
Y


# apply DBSCAN at every timestamps and creating clusters based on the Euclidean distances
allvalues = []
#eps_values =  np.arange(0.1, 3.1, 0.1)
eps_values =  np.arange(0.9, 2.1, 0.1)
#eps_values =  np.arange(0.9, 2.1, 0.01) 
iou_values = []
matched_dict = {}
for i in tqdm(eps_values):
    print("eps__value",i)
    clusters = np.array([])
    a = X.shape[1]//2
    for ik in range(X.shape[0]):
        clustering = DBSCAN(eps = i, min_samples = 2).fit(Y[a*ik:a*(ik+1),:])
        clusters = np.append(clusters,np.array(clustering.labels_))
    clusters = clusters.reshape(X.shape[0],a)
    clusters
    
    cluster_df = pd.DataFrame(clusters,columns =agent_ids,index=frame_id)
    
    cluster_df["max_groups"] = cluster_df.max(axis=1)
    
    
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
         #   indvidual_groups.update({index: groups})
            
            
    ##### Calculating the Co-existing time ratio here
   # rvalues = np.linspace(0.1,1,20)
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
        #print("loop is coming here")        
        #Calculating the IOU mteric here
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
            
        
for i in allvalues:
    if i[2]==max(iou_values):
        print("iou is maximum at the eps_value = "+str(i[0])+" and corelation_value = "+str(i[1])+" with value",i[2])

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

##iou is maximum at the eps_value = 1.5700000000000007 and 
#corelation_value = 0.5099999999999998 with value 0.5753424657534246

#iou is maximum at the eps_value = 1.5 and corelation_value = 0.43999999999999984 
#with value 0.573170731707317
#iou_max_at = np.where(ious == np.amax(ious))
#print("Hence, we say that iou is maximum at esp =",esp_values[iou_max_at])






