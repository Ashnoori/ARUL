# ARUL
Attention-based Remaining Useful Lifetime (ARUL) prediction for Aero-propulsion Systems - LIKE 2022

Citation: Noori, Arash, et al. “Attention-Based Remaining Useful Lifetime (ARUL) Prediction for Aero-Propulsion Systems.” LIKE Global, LIKE (Location Intelligence Knowledge Exchange), 14 June 2022. 

https://www.like-global.com/_files/ugd/04e87e_4e9cce3a4f9347a0ba7262130274affc.pdf.

Remaining Useful Lifetime (RUL) prediction is a key component in Prognostics and Health Management (PHM) of aircraft and aeroengines. Accurate and real-time RUL prediction can increase the safety of aircraft and decrease operation and maintenance costs. Previous approaches employ variants of recurrent neural network architectures (e.g., LSTM) to encode sensor data at each time step. In other domains of artificial intelligence, such as Natural Language Processing (NLP), neural attention-based architectures have been shown to have superior performance compared to other available architectures in terms of capturing long-distance dependencies in time series data [17], which can play a critical role in predicting RUL. In this work, to learn the RUL relationships of continually degrading equipment pertinent to an aeroengine, we propose ARUL, a neural attention-based model to more efficiently capture the long-term characteristics in a sequence of sensor measurements. In the PHM domain, the development of prognostic models requires run-to-failure trajectories, which are rarely available in real safety-critical applications due to the low occurrence of failures and faults in these systems. To overcome this, the proposed method is applied to 4 datasets generated from the simulation of realistic large commercial turbofan engines. Specifically, turbofan simulation data obtained from the NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) is used to assess the performance of the proposed raw transformer architecture for estimating the RUL of turbofan engines. Additionally, we illustrated the benefits and drawbacks of different components of the Transformers architecture in the context of time-series prediction.

Dataset can be found: https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data/xaut-bemq

For CUDA enabled please refer to xxx_gpu_03.py
