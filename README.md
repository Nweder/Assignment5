# Assignment 5 – LiDAR Data Processing #
**Name:** Mohamad Nweder  
**Course:** Industrial AI and eMaintenance - Part I: Theories & Concepts D7015B 55031 VT2025 
## GitHub Repository
The full project (code, plots, and documentation) is available at:  
[https://github.com/Nweder/Assignment5](https://github.com/Nweder/Assignment5)
--------------------------------------------------------------------

## Outcome
- Processing of raw LiDAR data using Python
- Application of ML methods (DBSCAN) and parameter tuning
- Documentation of results with plots

--------------------------------------------------------------------
## Task 1 – Ground Level Detection

The ground level was found using a histogram of Z-values and selecting the peak representing the ground.

### Dataset 1
- Ground level: 61.250

![Ground Level Histogram Dataset1](plots/dataset1_hist.png)

### Dataset 2
- Ground level: 61.265

![Ground Level Histogram Dataset2](plots/dataset2_hist.png)

--------------------------------------------------------------------

## Task 2 – Optimal Epsilon for DBSCAN

![Reslut picture](image.png)

The optimal `eps` value was determined using the elbow (k-distance) method.

### Dataset 1
- Estimated optimal eps: 0.155  
- Number of clusters: 799

**Elbow plot:**
![Elbow Plot Dataset1](plots/dataset1_elbow.png)

**Cluster plot:**
![Cluster Plot Dataset1](plots/dataset1_clusters.png)

--------------------------------------------------------------------

### Dataset 2
- Estimated optimal eps: 0.132 
- Number of clusters: 1180

**Elbow plot:**
![Elbow Plot Dataset2](plots/dataset2_elbow.png)

**Cluster plot:**
![Cluster Plot Dataset2](plots/dataset2_clusters.png)

--------------------------------------------------------------------

## Task 3 – Largest Cluster (Catenary)

![Reslut picture](image.png)

The largest cluster (excluding noise) was identified as the catenary.  
Bounding box coordinates (min/max X and Y) are reported below.

### Dataset 1
- **Bounds:**

--------------------------------------------------------------------