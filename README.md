**Temporal difference** is a method used to estimate the value function of a strategy.

It combines the ideas of Monte Carlo and dynamic programming algorithms:

* Similarities between Temporal difference method and **Monte Carlo**: Can learn from sample data without prior knowledge of the environment.
* Similarities between Temporal difference method and **Dynamic programming**: According to the idea of Bellman equation, the value estimate of the <u>current state</u> is updated by the value estimate of the <u>subsequent state</u>.

**Monte Carlo method**: Using <u>probability and statistical methods</u> to induce numerical estimates of the target we want from the sampling results.

State-value function based on strategy π in MDP :
【The value of <u>a single state</u> is its expected return】
$$V^\pi (s) = E_\pi [G_t | S_t = s]$$

However, MDP will have many state directions, so sampling on the MDP chain with a policy will have many sequences, which can be set to N sequences.
$$V^\pi (s) \times N = E_\pi [G_t | S_t = s] \times N$$
$$ E_\pi [G_t | S_t = s] \times N \gets G_1 + G_2 + G_3 + ... + G_{t-1} + G_t$$
$$V^\pi (s) = E_\pi [G_t | S_t = s] \approx \frac{1}{N} \sum_{N}^{i = 1} G_{t}^{(i)}$$

Monte Carlo method for the <u>incremental update</u> of the value function:
$$
V(s_t) \gets V(s_{t-1}) + \alpha [G_{t-1} - V(s_{t-1})]
$$

## Q_learning robot grib
![](TemporalDifference/robot_move.png)

## sarsa raward 
![](TemporalDifference/Sarsa_grid_CumulativeReward.png)

## n步Sarsa算法
![](TemporalDifference/nStep_Sarsa.png)

## Q_learning
![](TemporalDifference/Q_learning_cumulativeReward.png)