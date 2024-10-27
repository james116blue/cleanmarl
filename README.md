 # CleanMARL (Clean Implementation of MARL Algorithms)
 
Based on philosophy of this project: 
* [CleanRL](https://github.com/vwxyzjn/cleanrl/)

CleanMARL is a Deep MultiAgent Reinforcement Learning library that provides high-quality single-file implementation with research-friendly features. 

# Algorithms Implemented


| Algorithm                                                                         | Variants Implemented                                                                                                                         |
|-----------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| ✅ [MultiAgent Proximal Policy Gradient (MAPPO)](https://arxiv.org/pdf/2103.01955) | [`mappo_mpe.py`](https://github.com/james116blue/cleanmarl/blob/master/cleanmarl/mappo_mpe.py) |
|                                                                                   |  
##### Implementation features
1. Env state based on concatanation of local observations 
2. Huber loss for critic (value) network
3. Value normalization 

Authors don`t elaborate on math related to value normalization, but actualy it was done in the next manner (clip on minimum value to exclude zeros omitted)
```math
``\text{mean} = \mathop{\mathbb{E}}[R]
```
```math
\text{mean_sq} = \mathop{\mathbb{E}}[R^2]$$
```
```math
\beta\text{-debiasing term } 
```
```math
\text{mean}_t = w*\text{mean}_{t-1} + (1-w)*\text{minibatch.mean()}
```
```math
\text{mean_sq}_t =w*\text{mean}_{t-1} + (1-w)*\text{minibatch.mean()}^2
```
```math
\beta_t=w\beta_{t-1} + (1-w)*1
```
```math
v_\text{normalized} = \frac{v - \text{mean}/\beta}{\text{mean_sq}/\beta- \text{mean}^2}$$
```