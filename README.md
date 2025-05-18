[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![PR's Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](http://makeapullrequest.com)
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

# 🏄‍♂️ Awesome-Trajectory-Data-Management-and-Mining

Welcome to our carefully curated collection of **Deep Learning Methods and Foundation Models (LLM, LM, FM) for Trajectory Computing (Trajectory Data Management and Mining)** with awesome resources (paper, code, data, tool, etc.)! This repository serves as a valuable addition to our comprehensive survey paper. Rest assured, we are committed to consistently updating it to ensure it remains up-to-date and relevant.

<img src="./Picture/Trajectory_Overview.gif" width = "900" align=center>

By [Citymind LAB](https://citymind.top) <img src="./Picture/citymind.png" alt="图标" style="width: 108px; height: 20px;">, [HKUST(GZ)](https://www.hkust-gz.edu.cn/) <img src="./Picture/hkust-gz.png" alt="图标" style="width: 20px; height: 20px;">.

Check out our comprehensive tutorial paper:
> *Deep Learning for Trajectory Data Management and Mining: A Survey and Beyond.* <br/> Wei Chen, Yuxuan Liang†, Yuanshao Zhu, Yanchuan Chang, Kang Luo, Haomin Wen, Lei Li, Yanwei Yu, Qingsong Wen, Chao Chen, Kai Zheng, Yunjun Gao, Xiaofang Zhou, Fellow, IEEE, Yu Zheng, Fellow, IEEE. [[Link](https://arxiv.org/pdf/2403.14151.pdf)]

> **<p align="justify"> Abstract:** Trajectory computing is a pivotal domain encompassing trajectory data management and mining, garnering widespread attention due to its crucial role in various practical applications such as location services, urban traffic, and public safety. Traditional methods, focusing on simplistic spatio-temporal features, face challenges of complex calculations, limited scalability, and inadequate adaptability to real-world complexities. In this paper, we present a comprehensive review of the development and recent advances in deep learning for trajectory computing (DL4Traj). We first define trajectory data and provide a brief overview of widely-used deep learning models. Systematically, we explore deep learning applications in trajectory management (pre-processing, storage, analysis, and visualization) and mining (trajectory-related forecasting, trajectory-related recommendation, trajectory classification, travel time estimation, anomaly detection, and mobility generation). Notably, we encapsulate recent advancements in Large Language Models (LLMs) that hold the potential to augment trajectory computing. Additionally, we summarize application scenarios, public datasets, and toolkits. Finally, we outline current challenges in DL4Traj research and propose future directions. </p>



***We strongly encourage authors of relevant works to make a pull request and add their paper's information [[here](https://github.com/yoshall/Awesome-Trajectory-Computing/pulls)].***

👉 If you find any missed resources (paper / code / dataset / tool) or errors, please feel free to open an issue or make a pull request 🫡.

👉 Please consider giving this repository a star ⭐ if you find it helpful!


____

## 📰 News
```
- 2024.03.19: Successful launch of DL4Traj project! 😊
```

____

## 📚 Citation

👉 If you find our work useful in your research, please consider citing 👻:
```
@article{chen2024deep,
      title={Deep Learning for Trajectory Data Management and Mining: A Survey and Beyond}, 
      author={Wei Chen, Yuxuan Liang, Yuanshao Zhu, Yanchuan Chang, Kang Luo, Haomin Wen, Lei Li, Yanwei Yu, Qingsong Wen, Chao Chen, Kai Zheng, Yunjun Gao, Xiaofang Zhou, Yu Zheng},
      journal={arXiv preprint arXiv:2403.14151},
      year={2024}
}
```

____

## 📇 Contents

- [Related Surveys](#📖-related-surveys)
- [Taxonomy Framework](#🖲️-taxonomy-framework)
    - [Trajectory Data Management Paper List](#deep-learning-for-trajectory-data-management-🔐)
    - [Trajectory Data Mining Paper List](#deep-learning-for-trajectory-data-mining-🔍)
- [Summary of Resources](#summary-of-resources-🛠️)
    - [Datasets](#datasets)
    - [Tools](#tools)
    - [Other Useful Links](#other-useful-links)

____

## 📖 Related Surveys
- Computing with Spatial Trajectories [[paper]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/TrajectoryComputing_Preview.pdf)
- Trajectory data mining: an overview [[paper](https://dl.acm.org/doi/pdf/10.1145/2743025)]
- A survey on trajectory data mining: Techniques and applications [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7452339)]
- Trajectory data mining: A review of methods and applications [[paper](https://digitalcommons.library.umaine.edu/cgi/viewcontent.cgi?article=1084&context=josis)]
- A survey on trajectory clustering analysis [[paper](https://arxiv.org/pdf/1802.06971.pdf)]
- Trajectory data classification: A review [[paper](https://dl.acm.org/doi/pdf/10.1145/3330138)]
- A comprehensive survey on trajectory-based location prediction [[paper](https://link.springer.com/article/10.1007/s42044-019-00052-z)]
- A survey on trajectory data management, analytics, and learning [[paper](https://dl.acm.org/doi/pdf/10.1145/3440207)]
- A survey on deep learning for human mobility [[paper](https://dl.acm.org/doi/pdf/10.1145/3485125)]
- Classifying spatial trajectories [[paper](https://arxiv.org/pdf/2209.01322.pdf)]
- Traffic prediction using artificial intelligence: review of recent advances and emerging opportunities [[paper](https://www.sciencedirect.com/science/article/pii/S0968090X22003345?casa_token=G8k5h8mgbPoAAAAA:F9dJ-LJlP0WhaY83IIeuKbMb2rbUVsl7ZS_EI2x-ynqclVFUvUxvR3aLO9wKmuJtY8KOoN3r2uo)]
- A benchmark of existing tools for outlier detection and cleaning in trajectories [[paper](https://www.researchsquare.com/article/rs-3356633/v1)]
- Spatio-temporal trajectory similarity measures: A comprehensive survey and quantitative study [[paper](https://www.computer.org/csdl/journal/tk/5555/01/10275094/1R8pcQ2Casw)]
- Trajectory similarity measurement: An efficiency perspective [[paper](https://arxiv.org/pdf/2311.00960.pdf)]
- MobilityDL: A review of deep learning from trajectory data [[paper](https://arxiv.org/pdf/2402.00732.pdf)]
- Spatio-temporal data mining: A survey of problems and methods [[paper](https://dl.acm.org/doi/pdf/10.1145/3161602)]
- Deep Learning for Spatio-Temporal Data Mining: A Survey [[paper](https://dl.acm.org/doi/pdf/10.1145/3161602)]

<img src="./Picture/Survey_Compare.png" width = "900" align=center>

____

## 🖲️ Taxonomy Framework


**This survey is structured along follow dimensions:** 

> * [Deep Learning for Trajectory Data Management](#deep-learning-for-trajectory-data-management)
>    * [Pre-Processing](#pre-processing)
>        * [Simplification](#simplification)
>        * [Recovery](#recovery)
>        * [Map-Matching](#map-matching)
>    * [Storage](#storage)
>        * [Storage Database](#storage)
>        * [Index & Query](#index--query)
>    * [Analytics](#analytics)
>        * [Similarity Measurement](#similarity-measurement)
>        * [Cluster Analysis](#cluster-analysis)
>    * [Visualization](#visualization)
>    * [Recent advances in LLMs for trajectory management](#recent-advances-in-llms-for-trajectory-management)
>* [Deep Learning for Trajectory Data Mining](#deep-learning-for-trajectory-data-mining)
>    * [Trajectory-related Forecasting](#trajectory-related-forecasting)
>        * [Location Forecasting](#location-forecasting)
>        * [Traffic Forecasting](#traffic-forecasting)
>    * [Trajectory-related Recommendation](#trajectory-related-recommendation)
>        * [Travel Recommendation](#travel-recommendation)
>        * [Friend Recommendation](#friend-recommendation)
>    * [Trajectory-related Classification](#trajectory-related-classification)
>        * [Travel Mode Identification](#travel-mode-identification)
>        * [Trajectory-User Linking](#trajectory-user-linking)
>        * [Other Perspectives](#other-perspectives)
>    * [Travel Time Estimation](#travel-time-estimation)
>        * [Trajectory-based](#trajectory-based)
>        * [Road-based](#road-based)
>        * [Other Perspectives](#other-perspectives-1)
>    * [Anomaly Detection](#anomaly-detection)
>        * [Offline Detection](#offline-detection)
>        * [Online Detection](#online-detection)
>    * [Mobility Generation](#mobility-generation)
>        * [Macro-dynamic](#macro-dynamic)
>        * [Micro-dynamic](#micro-dynamic)
>    * [Multi-modal Trajectory Reserach](#multimodality)

>    * [Recent advances in LLMs for trajectory mining](#recent-advances-in-llms-for-trajectory-mining)


<img src="./Picture/Taxonomy.png" width = "800" align=center>

____

## **Deep Learning for Trajectory Data Management 🔐**


<details>
<summary>Pre-Processing</summary>

<img src="./Picture/pre-process.png" width = "500" align=center>

### <font color=Orange>Simplification</font>

- Traditional Methods
    - Batch Mode
        - Algorithms for the Reduction of the Number of Points Required to Represent a Digitized Line or its Caricature
        - Direction-preserving  trajectory simplification
    - Online Mode
        - An online algorithm for segmenting time series
        - Spatiotemporal compression techniques for moving point objects
    - Semantic-based
        - Trace: Realtime compression of streaming trajectories in road networks
- Deep Learning Methods
    - Trajectory simplification with reinforcement learning
    - A Lightweight Framework for Fast Trajectory Simplification
    - Error-bounded Online Trajectory Simplification with Multi-agent Reinforcement Learning
    - Collectively simplifying trajectories in a database: A query accuracy driven approach


### <font color=Orange>Recovery</font>

- Traditional Methods
    - Kinematic interpolation of movement data
    - A comparison of two methods to create tracks of moving objects: linear weighted distance and constrained random walk
    - Interpolation of animal tracking data in a fluid environment
- Deep Learning Methods
    - Free-space based
        - Deep trajectory recovery with fine-grained calibration using kalman filter
        - Attnmove: History enhanced trajectory recovery via attentional network
        - Periodicmove: shift-aware human mobility recovery with graph neural network
        - Trajbert: Bert-based trajectory recovery with spatial-temporal refinement for implicit sparse trajectories
        - Teri: An effective framework for trajectory recovery with irregular time intervals
    - Non Free-space based
        - Mtrajrec: Map-constrained trajectory recovery via seq2seq multitask learning
        - Rntrajrec: Road network enhanced trajectory recovery with spatial-temporal transformer
        - Visiontraj: A noise-robust trajectory recovery framework based on large-scale camera network
        - Learning semantic behavior for human mobility trajectory recovery
        - Traj2traj: A road network constrained spatiotemporal interpolation model for traffic trajectory restoration
        - Patr: Periodicity-aware trajectory recovery for express system via seq2seq model
    - Road Networks
        - Learning to generate maps from trajectories
        - Multimodal deep learning for robust road attribute detection
        - Aerial images meet crowdsourced trajectories: a new approach to robust road extraction
        - Deepdualmapper: A gated fusion network for automatic map extraction using aerial images and trajectories
        - Df-drunet: A decoder fusion model for automatic road extraction leveraging remote sensing images and GPS trajectory data
        - Delvmap: Completing residential roads in maps based on couriers’ trajectories and satellite imagery

### <font color=Orange>Map-Matching</font>

- Traditional Methods
    - Road reduction filtering for GPS-gis navigation
    - A general map matching algorithm for transport telematics applications
    - Map-matching in complex urban road networks
    - Fast map matching, an algorithm integrating hidden markov model with precomputation
- Deep Learning Methods
    - Deepmm: Deep learning based map matching with data augmentation
    - Transformer-based mapmatching model with limited labeled data using transfer-learning approach
    - L2mm: learning to map matching with deep models for low-quality GPS trajectory data
    - Graphmm: Graph-based vehicular map matching by leveraging trajectory and road correlations
    - Dmm: Fast map matching for cellular data
    - Map-matching on wireless traffic sensor data with a sequence-to-sequence model
    - Fl-amm: Federated learning augmented map matching with heterogeneous cellular moving trajectories
</details>

____

<details>
<summary>Storage</summary>

### <font color=Orange>Storage Database</font>

- Trajectory Management Systems
    - Sharkdb: An in-memory column-oriented trajectory storage
    - Elite: an elastic infrastructure for big spatiotemporal trajectories
    - Dragoon: a hybrid and efficient big trajectory management system for offline and online analytics
    - Trajmesa: A distributed nosql-based trajectory data management system
- Vector Databases
    - Vector-based trajectory storage and query for intelligent transport system
    - Ghost: A general framework for high-performance online similarity queries over distributed trajectory streams

### <font color=Orange>Index & Query</font>

- Traditional index
    - Trajectory similarity join in spatial networks,
    - Distributed Trajectory Similarity Search
    - Distributed In-Memory Trajectory Similarity Search and Join on Road Network
    - Trass: Efficient trajectory similarity search based on key-value data stores
- Deep Learning Methods
    - Effectively Learning Spatial Indices
    - The Case for Learned Spatial Indexes
    - X-FIST: Extended Flood Index for Efficient Similarity Search in Massive Trajectory Dataset
</details>

____

<details>
<summary>Analytics</summary>

### <font color=Orange>Similarity Measurement</font>

<img src="./Picture/similar_compare.png" width = "600" align=center>

- Traditional Methods
    - Efficient retrieval of similar time sequences under time warping
    - Discovering Similar Multidimensional Trajectories
    - Robust and Fast Similarity Search for Moving Object Trajectories
    - Computing discrete fr ́ echet distance
    - The computational geometry of comparing shapes
- Deep Learning Methods
    - <img src="./Picture/similar_learning_compare.png" width = "800" align=center>
    - Free-Space
        - Supervised Learning-based
            - Computing Trajectory Similarity in Linear Time: A Generic Seed-guided Neural Netric learning approach
            - Trajectory Similarity Learning with Auxiliary Supervision and Optimal Matching
            - Embeddingbased Similarity Computation for Massive Vehicle Trajectory Data
            - TMN: Trajectory Matching Networks for Predicting Similarity
            - T3S: Effective Representation Learning for Trajectory Similarity Computation
            - TrajGAT: A Graph-based Long-term Dependency Modeling Approach for Trajectory Similarity Computation
        - Self-Supervised Learning-based
            - Deep Representation Learning for Trajectory Similarity Computation
            - Towards Robust Trajectory Similarity Computation: Representation-based Spatiotemporal Similarity Quantification
            - Similar Trajectory Search with Spatio-temporal Deep Representation Learning
            - Representation Learning with Multi-level Attention for Activity Trajectory Similarity Computation,
            - Efficient Trajectory Similarity Computation with Contrastive Learning
            - Trajectory Similarity Learning with Dual-Feature Attention
            - Self-supervised contrastive representation learning for large-scale trajectories
            - CSTRM: Contrastive Self-Supervised Trajectory Representation Model for Trajectory Similarity Computation
            - On Accurate Computation of Trajectory Similarity via Single Image Superresolution
            - Effective and Efficient Sports Play Retrieval with Deep Representation Learning
    - Road Network
        - Supervised Learning-based
            - A Graphbased Approach for Trajectory Similarity Computation in Spatial Networks
            - GRLSTM: Trajectory Similarity Computation with Graph-based Residual LSTM
            - Spatiotemporal Trajectory Similarity Learning in Road Networks
            - Spatial Structure-Aware Road Network Embedding via Graph Contrastive Learning
        - Self-Supervised Learning-based
            - Trembr: Exploring Road Networks for Trajectory Representation Learning
            - Lightpath: Lightweight and scalable path representation learning



### <font color=Orange>Cluster Analysis</font>

- Traditional Methods
    - A review of moving object trajectory clustering algorithms
- Deep Learning Methods
    - <img src="./Picture/cluster.png" width = "800" align=center>
    - Multi-Stage 
        - Trajectory clustering via deep representation learning
        - Trip2vec: a deep embedding approach for clustering and profiling taxi trip purposes
    - End-to-End
        - Deep trajectory clustering with autoencoders
        - Detect: Deep trajectory clustering for mobility-behavior analysis
        - E2dtc: An end to end deep trajectory clustering framework via self-training

</details>

____

<details>
<summary>Visualization</summary>


- Traditional Methods
    - A descriptive framework for temporal data visualizations based on generalized space-time cubes
    - A survey of urban visual analytics: Advances and future directions
- Deep Learning Methods
    - Group Visualization
        - A visual analytics system for exploring, monitoring, and forecasting road traffic congestion
        - Visual abstraction of large scale geospatial origin-destination movement data
    - Individual Visualization
        - Deep learning-assisted comparative analysis of animal trajectories with deephl
        - Visualization of driving behavior based on hidden feature extraction by using deep learning
        - Deep learning detection of anomalous patterns from bus trajectories for traffic insight analysis

</details>

____

<details>
<summary>Recent advances in LLMs for trajectory management</summary>

- Recovery & Enhancement
    - Spatio-temporal storytelling? leveraging generative models for semantic trajectory analysis
    - An exploratory assessment of llm’s potential toward flight trajectory reconstruction analysis

</details>

____

## **Deep Learning for Trajectory Data Mining 🔍**

____


<details>
<summary>Trajectory-related Forecasting</summary>

<img src="./Picture/forecasting.png" width = "600" align=center>

### Location Forecasting

- Deep Learning Methods
    - Next Location Prediction
        - Deepmove: Predicting human mobility with attentional recurrent networks
        - Predicting human mobility via variational attention
        - Deeptransport: Prediction and simulation of human mobility and transportation mode at a citywide level
        - Location prediction over sparse user mobility traces using rnns
        - Mobtcast: Leveraging auxiliary trajectory forecasting for human mobility prediction
        - Artificial neural networks applied to taxi destination prediction
        - A bilstm-cnn model for predicting users’ next locations based on geotagged social media
        - A neural network approach to jointly modeling social networks and mobile trajectories
        - Context-aware deep model for joint mobility and time prediction
        - Serm: A recurrent model for next location prediction in semantic trajectories
        - Predicting the next location: A recurrent model with spatial and temporal contexts
        - Hst-lstm: A hierarchical spatial-temporal long-short term memory network for location prediction
        - Recurrent marked temporal point processes: Embedding event history to vector
        - Pre-training context and time aware location embeddings from spatial-temporal trajectories for user next location prediction
        - Mcn4rec: Multi-level collaborative neural network for next location recommendation
        - Taming the Long Tail in Human Mobility Prediction
    - Next POI Recommendation
        - A survey on deep learning based point-of-interest (poi) recommendations
        - Point-of-interest recommender systems based on location-based social networks: a survey from an experimental perspective
    - Unfinished Route Prediction
        - A survey on service route and time prediction in instant delivery: Taxonomy, progress, and prospects
        - Package pick-up route prediction via modeling couriers’ spatial-temporal behaviors
        - Graph2route: A dynamic spatial-temporal graph neural network for pick-up and delivery route prediction

____

### Traffic Forecasting

- Traditional Methods
    - Space–time modeling of traffic flow
    - Vector autoregressive models: specification, estimation, inference, and forecasting
    - Urban flow prediction from spatiotemporal data using machine learning: A survey
    - Recent trends in crowd analysis: A review,” Machine Learning with Applications
- Deep Learning Methods
    - Spatio-Temporal Grid
        - Deep spatio-temporal residual networks for citywide crowd flows prediction
        - Deep multi-view spatial-temporal network for taxi demand prediction
        - Spatio-temporal recurrent convolutional networks for citywide short-term crowd flows prediction
        - Periodic-crn: A convolutional recurrent model for crowd density prediction with recurring periodic patterns
        - Deepcrowd: A deep model for large-scale citywide crowd density and flow prediction
        - Urbanfm: Inferring fine-grained urban flows
        - Revisiting spatial-temporal similarity: A deep learning framework for traffic prediction
        - Deepurbanevent: A system for predicting citywide crowd dynamics at big events
        - Promptst: Prompt-enhanced spatio-temporal multi-attribute prediction
        - Multiattention 3d residual neural network for origin-destination crowd flow prediction
        - When transfer learning meets crosscity urban flow prediction: spatio-temporal adaptation matters
    - Spatio-Temporal Graph
        - Predicting citywide crowd flows in irregular regions using multi-view graph convolutional networks
        - Temporal multiview graph convolutional networks for citywide traffic volume inference
        - Mdtp: A multisource deep traffic prediction framework over spatio-temporal trajectory data

</details>

____


<details>
<summary>Trajectory-related Recommendation</summary>

<img src="./Picture/LBSN.png" width = "600" align=center>

### Travel Recommendation

- Traditional Methods
    - A survey of route recommendations: Methods, applications, and opportunities
- Deep Learning Methods
    - Hybrid Type
        - Learning effective road network representation with hierarchical graph neural networks
    - Sequential-based
        - Ldferr: A fuel-efficient route recommendation approach for long-distance driving based on historical trajectories
        - Progrpgan: Progressive gan for route planning
        - Personalized path recommendation with specified way-points based on trajectory representations
        - Personalized long distance fuel-efficient route recommendation through historical trajectories mining
        - Query2trip: Dual-debiased learning for neural trip recommendation
    - Graph-based
        - Dual-grained human mobility learning for location-aware trip recommendation with spatial–temporal graph knowledge fusion
        - Learning improvement heuristics for solving routing problems
        - Learning to effectively estimate the travel time for fastest route recommendation
    - Multi-modal Type
        - Walking down a different path: route recommendation based on visual and facility based diversity
        - Multi-modal transportation recommendation with unified route representation  learning
    - Reinforcement Learning
        - Spatio-temporal feature fusion for dynamic taxi route recommendation via deep reinforcement learning
        - Evacuation route recommendation using auto-encoder and markov decision process

____

### Friend Recommendation

- Traditional Methods
    - Recommendations in location-based social networks: a survey
    - Where online friends meet: Social communities in location-based networks
    - Friend recommendation for location-based mobile social networks
    - Geo-friends recommendation in gps-based cyber-physical social network
- Deep Learning Methods
    - Revisiting user mobility and social relationships in lbsns: a hypergraph embedding approach
    - Lbsn2vec++: Heterogeneous hypergraph embedding for location-based social networks
    - Social link inference via multi view matching network from spatiotemporal trajectories
    - Trajectory-based social circle inference
    - Graph structure learning on user mobility data for social relationship inference
    - Friend recommendation in location based social networks via deep pairwise learning

</details>

____

<details>
<summary>Trajectory-related Classification</summary>

<img src="./Picture/classification.png" width = "600" align=center>

- Traditional Methods
    - A survey and comparison of trajectory classification methods
    - Traclass: trajectory classification using hierarchical region-based and trajectory-based clustering
    - Integrating cross-scale analysis in the spatial and temporal domains for classification of behavioral movement
    - Learning transportation mode from raw gps data for geographic applications on the web
    - Revealing the physics of movement: Comparing the similarity of movement characteristics of different types of moving objects

### Travel Mode Identification

- Deep Learning Methods
    - Estimator: An effective and scalable framework for transportation mode classification over trajectories
    - Trajectorynet: An embedded gps trajectory representation for point-based classification using recurrent neural networks
    - Spatio-temporal gru for trajectory classification
    - Modeling trajectories with neural ordinary differential equations
    - Traclets: Harnessing the power of computer vision for trajectory classification
    - Trajformer: Efficient trajectory classification with transformers
    - Semi-supervised deep learning approach for transportation mode identification using gps trajectory data
    - Distributional and spatial-temporal robust representation learning for transportation activity recognition
    - End-to-end trajectory transportation mode classification using bi-lstm recurrent neural network
    - Trajectory-as-a-sequence: A novel travel mode identification framework
    - A multi-scale attributes attention model for transport mode identification
    - Graph based embedding learning of trajectory data for transportation mode recognition by fusing sequence and dependency relations


____


### Trajectory-User Linking

- Deep Learning Methods
    - Identifying human mobility via trajectory embeddings
    - Trajectory-user linking via variational autoencoder
    - Trajectory-user linking with attentive recurrent network
    - Mutual distillation learning network for trajectory-user linking
    - Adversarial mobility learning for human trajectory classification
    - Self-supervised human mobility learning for next location prediction and trajectory classification

____

### Other Perspectives

- Deep Learning Methods
    - Semi-Supervised Learning
        - Semi-supervised deep learning approach for transportation mode identification using gps trajectory data
        - Semi-supervised deep ensemble learning for travel mode identification
        - Semi-supervised federated learning for travel mode identification from gps trajectories
    - Unsupervised Learning
        - Unsupervised deep learning for gps based transportation mode identification
    - Limited Data Scenario
        - Improving transportation mode identification with limited gps trajectories
        - A framework of travel mode identification fusing deep learning and map-matching algorithm
    - Unlabeled Data Scenario
        - S2tul: A semisupervised framework for trajectory-user linking
    - Cross-Platform Scenarios
        - Dplink: User identity linkage via deep neural network from heterogeneous mobility data
        - Egomuil: Enhancing spatio-temporal user identity linkage in location-based social networks with ego-mo hypergraph
    - Different Data Type Scenarios
        - Trajectory-user linking via hierarchical spatio-temporal attention networks

</details>

____


<details>
<summary>Travel Time Estimation</summary>

<img src="./Picture/estimation.png" width = "600" align=center>

### Traditional Methods
- 
    - Historical data based real time prediction of vehicle arrival time
    - A simple baseline for travel time estimation using large-scale trip data
    - HTTP: A new framework for bus travel time prediction based on historical trajectories
    - Route travel time estimation using low-frequency floating car data


### Deep Learning Methods 

- Trajectory-based
    - When will you arrive? estimating travel time based on deep neural networks
    - Traffic speed prediction and congestion source exploration: A deep learning method
    - Deeptravel: a neural network based travel time estimation model with auxiliary supervision
    - Multi-task representation learning for travel time estimation
    - Context-aware road travel time estimation by coupled tensor decomposition based on trajectory data
    - MTLM: a multi-task learning model for travel time estimation
    - TTPNet: A neural network for travel time prediction based on tensor decomposition and graph embedding
    

- Road-based
    - Learning to estimate the travel time
    - CoDriver ETA: Combine driver information in estimated time of arrival by driving style learning auxiliary task
    - DeepIST: Deep image-based spatiotemporal network for travel time estimation
    - CompactETA: A fast inference system for travel time prediction
    - Road network metric learning for estimated time of arrival
    - Interpreting trajectories from multiple views: A hierarchical selfattention network for estimating the time of arrival
    - Route travel time estimation on a road network revisited: Heterogeneity, proximity, periodicity and dynamicity
    - Graph
        - ConstGAT: Contextual spatial-temporal graph attention network for travel time estimation at baidu maps
        - HetETA: Heterogeneous information network embedding for estimating time of arrival
        - Dual graph convolution architecture search for travel time estimation

    


- Other Perspectives
    - En route
        - SSML: Self-supervised meta-learner for en route travel time estimation at baidu maps
        - MetaER-TTE: An Adaptive Meta-learning Model for En Route Travel Time Estimation
    - Uncertainty
        - Privacy-preserving travel time prediction with uncertainty using GPS trace data
        - Cross-area travel time uncertainty estimation from trajectory data: a federated learning approach
    - Classification-based
        - CatETA: A categorical approximate approach for estimating time of arrival
        - Uncertainty-aware probabilistic travel time prediction for on-demand ride-hailing at Didi
    - Travel time distribution
        - Citywide estimation of travel time distributions with bayesian deep graph learning
        - Travel time distribution estimation by learning representations over temporal attributed graphs

</details>

____


<details>
<summary>Anomaly Detection</summary>

- Traditional Methods
    - Trajectory outlier detection: A Partition-and-Detect Framework

### Offline Detection

- Deep Learning Methods
    - Anomalous Trajectory Detection using Recurrent Neural Network
    - Coupled igmmgans with applications to anomaly detection in human mobility data
    - TripSafe: Retrieving Safety-related Abnormal Trips in Real-time with Trajectory Data
    - Open anomalous trajectory recognition via probabilistic metric learning
    

____

### Online Detection

- Deep Learning Methods
    - A Fast Trajectory Outlier Detection Approach via Driving Behavior Modeling
    - Online Anomalous Subtrajectory Detection on Road Networks with Deep Reinforcement Learning
    - Online Anomalous Trajectory Detection with Deep Generative Sequence Modeling
    - DeepTEA: Effective and Efficient Online Time-dependent Trajectory Outlier Detection

</details>

____


<details>
<summary>Mobility Generation</summary>

<img src="./Picture/generation.png" width = "600" align=center>

### Macro-dynamic

- Traditional Methods
    - A deep gravity model for mobility flows generation
    - Unraveling the origin of exponential law in intra-urban human mobility Deep Learning Methods
    - Citywide traffic flow prediction based on multiple gated spatio-temporal convolutional neural networks
    - Spatiotemporal scenario generation of traffic flow based on LSTM-GAN
    - TrafficGAN: Network-scale deep traffic prediction with generative adversarial nets
    - Traffic flow imputation using parallel data and generative adversarial networks
    - GANs based density distribution privacypreservation on mobility data
    - Deep multi-view spatial-temporal network for taxi demand prediction

____

### Micro-dynamic

- Deep Learning Methods
    - Grids-based
        - A nonparametric generative model for human trajectories
        - Wherenext: a location predictor on trajectory pattern mining
        - Deeptransport: Prediction and simulation of human mobility and transportation mode at a citywide level
        - Deepmove: Predicting human mobility with attentional recurrent networks
        - How do you go where? improving next location prediction by learning travel mode information using transformers
        - Simulating continuous-time human mobility trajectories
        - COLA: Crosscity mobility transformer for human trajectory simulation
        - Activity trajectory generation via modeling spatiotemporal dynamics
        - Learning to simulate daily activities via modeling dynamic human needs
        - Where would I go next? large language models as human mobility predictors
        - Exploring large language models for human mobility prediction under public events
        - TrajGAIL: Generating urban vehicle trajectories using generative adversarial imitation learning
        - TrajGDM: A New Trajectory Foundation Model for Simulating Human Mobility
    - GPS-based
        - Large scale GPS trajectory generation using map based on two stage GAN
        - Generating mobility trajectories with retained data utility
        - Difftraj: Generating GPS trajectory with diffusion probabilistic model
        - ControlTraj: Controllable Trajectory Generation with Topology-Constrained Diffusion Model
        - Holistic Semantic Representation for Navigational Trajectory Generation
        - Seed: Bridging Sequence and Diffusion Models for Road Trajectory Generation
        - MA2Traj: Diffusion network with multi-attribute aggregation for trajectory generation
        - GTG: Generalizable Trajectory Generation Model for Urban Mobility
        - ST-DiffTraj: A Spatiotemporal-Aware Diffusion Model for Trajectory Generation
        - DiffPath: Generating Road Network based Path with Latent Diffusion Model
        - Map2Traj: Street Map Piloted Zero-shot Trajectory Generation with Diffusion Model

        
    
    
    

</details>

____

<details>
<summary>Multi-modal Trajectory Reserach</summary>

### Multi-modal Trajectory Retrieval
- Learning Generalized and Flexible Trajectory Models from Omni-Semantic Supervision

</details>

____

<details>
<summary>Recent advances in LLMs for trajectory mining</summary>

- Forecasting
    - Where would i go next? large language models as human mobility predictors
    - Exploring large language models for human mobility prediction under public events
    - Prompt mining for language-based human mobility forecasting
    - Urbangpt: Spatio-temporal large language models
- Generation
    - Large language models as urban residents: An llm agent framework for personal mobility generation
    - Mobilitygpt: Enhanced human mobility modeling with a gpt model
    - Large Language Models as Urban Residents: An LLM Agent Framework for Personal Mobility Generation
- Identification
    - Are you being tracked? discover the power of zero-shot trajectory tracing with llms!

</details>

____

## **Summary of Resources 🛠️**

____

<details>
<summary>Datasets</summary>

<!--<img src="./Picture/datasets.png" width = "800" align=center>-->

<table>
  <tr>
    <th>Data Category</th>
    <th>Type</th>
    <th>Dataset Name</th>
    <th>Main Area</th>
    <th>Duration </th>
   <th>Link</th>
  </tr>
  <tr>
    <td rowspan="16">Continuous GPS traces</td>
    <td>Human</td>
    <td>GeoLife</td>
    <td>Asia</td>
    <td>4.5 Years</td>
    <td><a href="https://www.microsoft.com/en-us/download/details.aspx?id=52367" target="_blank">link</a></td>
  </tr>
  <tr>
    <td>Human</td>
    <td>TMD</td>
    <td>Italiana</td>
    <td>31 Hours</td>
    <td><a href="https://cs.unibo.it/projects/us-tm2017/index.html" target="_blank">link</a></td>
  </tr>
  <tr>
    <td>Human</td>
    <td>SHL</td>
    <td>U.K.</td>
    <td>7 Months</td>
    <td><a href="http://www.shl-dataset.org/" target="_blank">link</a></td>
  </tr>
  <tr>
    <td>Human</td>
    <td>OpenStreetMap</td>
    <td>Global</td>
    <td>From 2005</td>
    <td><a href="https://www.openstreetmap.org/traces" target="_blank">link</a></td>
  </tr>
  <tr>
    <td>Human</td>
    <td>MDC</td>
    <td>Switzerland</td>
    <td>3 Years</td>
    <td><a href="https://www.idiap.ch/en/scientific-research/data/mdc" target="_blank">link</a></td>
  </tr>
  <tr>
    <td>Taxi</td>
    <td>T-Drive</td>
    <td>Beijing, China</td>
    <td>1 Week</td>
    <td><a href="https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/" target="_blank">link</a></td>
  </tr>
  <tr>
    <td>Taxi</td>
    <td>Porto</td>
    <td>Porto, Portugal</td>
    <td>9 Months</td>
    <td><a href="https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data" target="_blank">link</a></td>
  </tr>
  <tr>
    <td>Taxi</td>
    <td>Taxi-Shanghai</td>
    <td>Shanghai, China</td>
    <td>1 Year</td>
    <td><a href="https://cse.hkust.edu.hk/scrg" target="_blank">link</a></td>
  </tr>
  <tr>
    <td>Taxi</td>
    <td>DiDi-Chengdu</td>
    <td>Chengdu, China</td>
    <td>1 Month</td>
    <td><a href="https://gaia.didichuxing.com" target="_blank">link</a></td>
  </tr>
  <tr>
    <td>Taxi</td>
    <td>DiDi-Xi’an</td>
    <td>Xi’an, China</td>
    <td>1 Month</td>
    <td><a href="https://gaia.didichuxing.com" target="_blank">link</a></td>
  </tr>
  <tr>
    <td>Truck</td>
    <td>Greek</td>
    <td>Athens, Greece</td>
    <td>-</td>
    <td><a href="http://isl.cs.unipi.gr/db/projects/rtreeportal/" target="_blank">link</a></td>
  </tr>
  <tr>
    <td>Hurricane</td>
    <td>HURDAT</td>
    <td>Atlantic</td>
    <td>151 Years</td>
    <td><a href="https://www.aoml.noaa.gov/hrd/hurdat/Data_Storm.html" target="_blank">link</a></td>
  </tr>
  <tr>
    <td>Delivery</td>
    <td>Grab-Posisi-L</td>
    <td>Southeast Asia</td>
    <td>1 Months</td>
    <td><a href="https://engineering.grab.com/grab-posisi" target="_blank">link</a></td>
  </tr>
  <tr>
    <td>Vehicle</td>
    <td>NGSIM</td>
    <td>USA</td>
    <td>45 Minutes</td>
    <td><a href="https://catalog.data.gov/dataset/next-generation-simulation-ngsim-vehicle-trajectories-and-supporting-data" target="_blank">link</a></td>
  </tr>
  <tr>
    <td>Animal</td>
    <td>Movebank</td>
    <td>Global</td>
    <td>Decades</td>
    <td><a href="https://www.movebank.org/cms/movebank-main" target="_blank">link</a></td>
  </tr>
  <tr>
    <td>Vessel</td>
    <td>Vessel Traffic</td>
    <td>USA</td>
    <td>9 Years</td>
    <td><a href="https://marinecadastre.gov/AIS/" target="_blank">link</a></td>
  </tr>
  <tr>
    <td rowspan="14">Check-In sequences</td>
    <td>Human</td>
    <td>Gowalla</td>
    <td>Global</td>
    <td>1.75 Years</td>
    <td><a href="https://snap.stanford.edu/data/loc-gowalla.html" target="_blank">link</a></td>
  </tr>
  <tr>
    <td>Human</td>
    <td>Brightkite</td>
    <td>Global</td>
    <td>30 Months</td>
    <td><a href="https://snap.stanford.edu/data/loc-Brightkite.html" target="_blank">link</a></td>
  </tr>
  <tr>
    <td>Human</td>
    <td>Foursquare-NY</td>
    <td>New York, USA</td>
    <td>10 Months</td>
    <td><a href="https://sites.google.com/site/yangdingqi/home/foursquare-dataset" target="_blank">link</a></td>
  </tr>
  <tr>
    <td>Human</td>
    <td>Foursquare-TKY</td>
    <td>Tokyo, Japan</td>
    <td>10 Months</td>
    <td><a href="https://sites.google.com/site/yangdingqi/home/foursquare-dataset" target="_blank">link</a></td>
  </tr>
  <tr>
    <td>Human</td>
    <td>Foursquare-Global</td>
    <td>Global</td>
    <td>18 Months</td>
    <td><a href="https://sites.google.com/site/yangdingqi/home/foursquare-dataset" target="_blank">link</a></td>
  </tr>
  <tr>
    <td>Human</td>
    <td>Weeplace</td>
    <td>Global</td>
    <td>7.7 Years</td>
    <td><a href="https://www.yongliu.org/datasets/" target="_blank">link</a></td>
  </tr>
  <tr>
    <td>Human</td>
    <td>Yelp</td>
    <td>Global</td>
    <td>15 Years</td>
    <td><a href="https://www.yelp.com/dataset/" target="_blank">link</a></td>
  </tr>
  <tr>
    <td>Human</td>
    <td>Instagram</td>
    <td>New York, USA</td>
    <td>5.5 Years</td>
    <td><a href="https://dl.acm.org/doi/10.5555/3304222.3304226" target="_blank">paper</a></td>
  </tr>
  <tr>
    <td>Human</td>
    <td>GMove</td>
    <td>2 cities in USA</td>
    <td>20 Days</td>
    <td><a href="https://dl.acm.org/doi/10.1145/2939672.2939793" target="_blank">paper</a></td>
  </tr>
  <tr>
    <td>Taxi</td>
    <td>TLC</td>
    <td>New York, USA</td>
    <td>From 2009</td>
    <td><a href="https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page" target="_blank">link</a></td>
  </tr>
  <tr>
    <td>Bicycle</td>
    <td>Mobike-Shanghai</td>
    <td>Shanghai, China</td>
    <td>2 Weeks</td>
    <td><a href="https://www.heywhale.com/mw/dataset/5d315ebbcf76a60036e565bf">link</a></td>
  </tr>
  <tr>
    <td>Bicycle</td>
    <td>Bike-Xiamen</td>
    <td>Xiamen, China</td>
    <td>5 Days</td>
    <td><a href="https://data.xm.gov.cn/contest-series/digit-china-2021">link</a></td>
  </tr>
  <tr>
	<td>Bicycle</td>
    <td>Citi Bikes</td>
    <td>New York, USA</td>
    <td>From 2013</td>
    <td><a href="https://citibikenyc.com/system-data">link</a></td>
  </tr>
  <tr>
    <td>Delivery</td>
    <td>LaDe</td>
    <td>5 cities in China</td>
    <td>6 Months</td>
    <td><a href="https://wenhaomin.github.io/LaDe-website/">link</a></td>
  </tr>
  <tr>
    <td rowspan="2">Synthetic traces</td>
    <td>Taxi</td>
    <td>SynMob</td>
    <td>2 cities in China</td>
    <td>1 Month</td>
    <td><a href="https://yasoz.github.io/SynMob/">link</a></td>
  </tr>
  <tr>
    <td>Vehicle</td>
    <td>BerlinMod</td>
    <td>Berlin, German</td>
    <td>28 Days</td>
    <td><a href="https://secondo-database.github.io/BerlinMOD/BerlinMOD.html">link</a></td>
  </tr>
  <tr>
    <td rowspan="6">Other formats of trajectories</td>
    <td>Crowd Flow</td>
    <td>COVID19USFlows</td>
    <td>USA</td>
    <td>From 2019</td>
    <td><a href="https://github.com/GeoDS/COVID19USFlows">link</a></td>
  </tr>
  <tr>
    <td>Crowd Flow</td>
    <td>MIT-Humob2023</td>
    <td>Japan</td>
    <td>90 Days</td>
    <td><a href="https://connection.mit.edu/humob-challenge-2023">link</a></td>
  </tr>
  <tr>
    <td>Crowd Flow</td>
    <td>BousaiCrowd</td>
    <td>Japan</td>
    <td>4 Months</td>
    <td><a href="https://github.com/deepkashiwa20/DeepCrowd">link</a></td>
  </tr>
  <tr>
    <td>Traffic Flow</td>
    <td>TaxiBJ</td>
    <td>Beijing, China</td>
    <td>17 Months</td>
    <td><a href="https://github.com/amirkhango/DeepST">link</a></td>
  </tr>
  <tr>
    <td>Traffic Flow</td>
    <td>BikeNYC</td>
    <td>New York, USA</td>
    <td>6 Months</td>
    <td><a href="https://github.com/amirkhango/DeepST">link</a></td>
  </tr>
  <tr>
    <td>Traffic Flow</td>
    <td>TaxiBJ21</td>
    <td>Beijing, China</td>
    <td>3 Months</td>
    <td><a href="https://github.com/jwwthu/DL4Traffic/tree/main/TaxiBJ21">link</a></td>
  </tr>
</table>

> Here, we plan to host a Google Cloud Drive to collect all trajectory-related datasets for the convenience of researchers. (Coming soon🚀)

</details>

____

<details>
<summary>Tools</summary>

- [SUMO](https://eclipse.dev/sumo)
- [SafeGraph](https://docs.safegraph.com/docs/welcome)
- [Cblab](https://github.com/caradryanl/CityBrainLab)
- [PyTrack](https://github.com/titoghose/PyTrack)
- [PyMove](https://pymove.readthedocs.io/en/latest)
- [TransBigData](https://transbigdata.readthedocs.io/)
- [Traja](https://github.com/traja-team/traja)
- [MovingPandas](https://github.com/movingpandas/movingpandas)
- [Scikit-mobility](https://github.com/scikit-mobility/scikit-mobility)
- [Tracktable](https://github.com/sandialabs/tracktable)
- [Yupi](https://github.com/yupidevs/yupi)

</details>

____

<details>
<summary>Other Useful Links</summary>

- [Uber](https://www.uber.com)
- [DiDi](https://didiglobal.com)
- [Google Map](https://www.google.com/maps)
- [Baidu Map](https://map.baidu.com)
- [Cainiao](https://www.cainiao.com)

</details>

____


## 😍 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yoshall/Awesome-Trajectory-Computing&type=Date)](https://star-history.com/#yoshall/Awesome-Trajectory-Computing&Date)




