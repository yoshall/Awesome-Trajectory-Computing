[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![PR's Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](http://makeapullrequest.com)
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

# Awesome-Trajectory-Computing

Welcome to our carefully curated collection of **Deep Learning Methods and Foundation Models (LLM, LM, FM) for Trajectory Computing (Trajectory Data Mining and Management)** with awesome resources (paper, code, data, too, etc.)! This repository serves as a valuable addition to our comprehensive survey paper. Rest assured, we are committed to consistently updating it to ensure it remains up-to-date and relevant.

<img src="./Picture/Trajectory_Overview.gif" width = "900" align=center>

By [Citymind LAB](https://citymind.top)![alt citymind](./Picture/citymind.png), [HKUST(GZ)](https://www.hkust-gz.edu.cn/)![alt hkust-gz](./Picture/hkust-gz.png).

Check out our comprehsensive tutorial paper:
> *Deep Learning for Trajectory Data Management and Mining: A Survey and Beyond.* <br/> Wei Chen, Yuxuan Liang†, Yuanshao Zhu, Yanchuan Chang, Kang Luo, Haomin Wen, Lei Li, Yanwei Yu, Qingsong Wen, Chao Chen, Kai Zheng, Yunjun Gao, Xiaofang Zhou, Fellow, IEEE, Yu Zheng, Fellow, IEEE. [[Link](#)]

> **<p align="justify"> Abstract:** *Trajectory computing is a pivotal domain encompassing trajectory data management and mining, garnering widespread attention due to its crucial role in various practical applications such as location services, urban traffic, and public safety. Traditional methods, focusing on simplistic spatio-temporal features, face challenges of complex calculations, limited scalability, and inadequate adaptability to real-world complexities. In this paper, we present a comprehensive review of the development and recent advances in deep learning for trajectory computing (DL4Traj). We first define trajectory data and provide a brief overview of widely-used deep learning models. Systematically, we explore deep learning applications in trajectory management (pre-processing, storage, analysis, and visualization) and mining (forecasting, recommendation, classification, estimation, anomaly detection, and generation). Additionally, we
summarize application scenarios, public datasets, and toolkits. Finally, we outline current challenges in DL4Traj research and propose future directions.* </p>

***#### ***We strongly encourage authors of relevant works to make a pull request and add their paper's information [[here](https://github.com/yoshall/Awesome-Trajectory-Computing/pulls)].

👉 If you find any missed resources (paper/code) or errors, please feel free to open an issue 🫡 or make a pull request.

👉 Please consider giving this repository a star ⭐ if you find it helpful!


____

## News
```
- 2024.03.19: Latest update of this paper list.

```

## Citation

👉 If you find our work useful in your research, please consider citing 👻:
```
@misc{chen2024deep,
      title={Deep Learning for Trajectory Data Management and Mining: A Survey and Beyond}, 
      author={Wei Chen, Yuxuan Liang†, Yuanshao Zhu, Yanchuan Chang, Kang Luo, Haomin Wen, Lei Li, Yanwei Yu, Qingsong Wen, Chao Chen, Kai Zheng, Yunjun Gao, Xiaofang Zhou, Yu Zheng},
      year={2024},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Contents

- [Related Surveys](#related-surveys)
- [Taxonomy Framework](#taxonomy-framework)
    - [Trajectory Data Management Paper List](#deep-learning-for-trajectory-data-management)
    - [Trajectory Data Mining Paper List](#deep-learning-for-trajectory-data-mining)
- [Summary of Resources](#Taxonomy-and-summary-of-open-sourced-dataset)
    - [Datasets](#datasets)
    - [Tools](#tools)
    - [Other Useful Links](#other-useful-links)


## Related Surveys

- Trajectory data mining: an overview [[paper](#)]
- A survey on trajectory data mining: Techniques and applications [[paper](#)]
- Trajectory data mining: A review of methods and applications [[paper](#)]
- A survey on trajectory clustering analysis [[paper](#)]
- Trajectory data classification: A review [[paper](#)]
- A comprehensive survey on trajectory-based location prediction [[paper](#)]
- A survey on trajectory data management, analytics, and learning [[paper](#)]
- A survey on deep learning for human mobility [[paper](#)]
- Classifying spatial trajectories [[paper](#)]
- Traffic prediction using artificial intelligence: review of recent advances and emerging opportunities [[paper](#)]
- A benchmark of existing tools for outlier detection and cleaning in trajectories [[paper](#)]
- Spatio-temporal trajectory similarity measures: A comprehensive survey and quantitative study [[paper](#)]
- Trajectory similarity measurement: An efficiency perspective [[paper](#)]
- MobilityDL: A review of deep learning from trajectory data [[paper](#)]

<img src="./Picture/Survey_Compare.png" width = "900" align=center>


## Taxonomy Framework


**This survey is structured along follow dimensions:** 

* [Deep Learning for Trajectory Data Management](#deep-learning-for-trajectory-data-management)
    * [Pre-Processing](#pre-processing)
        * [Simplification](#simplification)
        * [Recovery](#recovery)
        * [Map-Matching](#map-matching)
    * [Storage](#storage)
        * [Storage Database](#storage)
        * [Index & Query](#index--query)
    * [Analytics](#analytics)
        * [Similarity Measurement](#similarity-measurement)
        * [Cluster Analysis](#cluster-analysis)
    * [Visualization](#visualization)
    * [Recent advances in LLMs for trajectory management](#recent-advances-in-llms-for-trajectory-management)
* [Deep Learning for Trajectory Data Mining](#deep-learning-for-trajectory-data-mining)
    * [Trajectory-related Forecasting](#trajectory-related-forecasting)
        * [Location Forecasting](#location-forecasting)
        * [Traffic Forecasting](#traffic-forecasting)
    * [Trajectory-related Recommendation](#trajectory-related-recommendation)
        * [Travel Recommendation](#travel-recommendation)
        * [Friend Recommendation](#friend-recommendation)
    * [Trajectory-related Classification](#trajectory-related-classification)
        * [Travel Mode Identification](#travel-mode-identification)
        * [Trajectory-User Linking](#trajectory-user-linking)
        * [Other Perspectives](#other-perspectives)
    * [Travel Time Estimation](#travel-time-estimation)
        * [Trajectory-based](#trajectory-based)
        * [Road-based](#road-based)
        * [Other Perspectives](#other-perspectives-1)
    * [Anomaly Detection](#anomaly-detection)
        * [Offline Detection](#offline-detection)
        * [Online Detection](#online-detection)
    * [Mobility Generation](#mobility-generation)
        * [Macro-dynamic](#macro-dynamic)
        * [Micro-dynamic](#micro-dynamic)
    * [Recent advances in LLMs for trajectory mining](#recent-advances-in-llms-for-trajectory-mining)
<img src="./Picture/Taxonomy.png" width = "800" align=center>


### Deep Learning for Trajectory Data Management

#### Pre-Processing

##### Simplification

coming soon

##### Recovery

coming soon

##### Map-Matching

coming soon

#### Storage

##### Storage Database

coming soon

##### Index & Query

coming soon

#### Analytics

##### Similarity Measurement

coming soon

##### Cluster Analysis

coming soon

#### Visualization

coming soon

#### Recent advances in LLMs for trajectory management

coming soon

### Deep Learning for Trajectory Data Mining

#### Trajectory-related Forecasting

##### Location Forecasting

coming soon

##### Traffic Forecasting

coming soon

#### Trajectory-related Recommendation

coming soon

##### Travel Recommendation

coming soon

##### Friend Recommendation

coming soon

#### Trajectory-related Classification

##### Travel Mode Identification

coming soon

##### Trajectory-User Linking

coming soon

##### Other Perspectives

coming soon

#### Travel Time Estimation

##### Trajectory-based

coming soon

##### Road-based

coming soon

##### Other Perspectives

coming soon

#### Anomaly Detection

##### Offline Detection

coming soon

##### Online Detection

coming soon

#### Mobility Generation

##### Macro-dynamic

coming soon

##### Micro-dynamic

coming soon

#### Recent advances in LLMs for trajectory mining

coming soon

## Summary of Resources

### Datasets

coming soon

### Tools

coming soon

### Other Useful Links

coming soon

