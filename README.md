# ppo-tris
Leonora Goble

## Overview
Tetris is a difficult problem for computers to solve. Part of the reason is the complexity of logic required for fitting pieces together in a way that minimizes gaps (see [Tetris is NP-hard](https://arxiv.org/pdf/2009.14336)) which is desirable if a player wishes to survive for a long time in the game and score by filling rows to clear them.

Another issue is the amount of time and planning required to acheive rewards by clearing lines. Clearing even a single line can take a minimum of 3-5 tetrominoes to be placed, and the ideal strategy of waiting until you can clear 4 lines at once to get a Tetris takes at least that much longer. Balancing this objective with the greater strategy of keeping the board as "neat" as possible (free of holes so lines can be cleared while minimizing height) provides the interest of the game, and is difficult to encode in a single reward function. 

Successful approaches to the problem used heuristic function approaches which measure features of the board such as the heights of the columns, maximum height of the board, number of holes in the board, and bumpiness (differences between column heights). Since in most Tetris games you are allowed to see the queue of upcoming tetrominoes, [minimax can also be used](https://github.com/ozhi/tetris-ai/tree/master), but these approaches require a specific treatment of the action space (grouping actions into piece placements rather than considering individual control inputs, discussed later). Q learning and deep Q learning have also been used to good effect.

I found an unofficial [Gymnasium environment for Tetris](https://github.com/Max-We/Tetris-Gymnasium) which I used for my project.

## Approach
I chose to use PPO (Proximal Policy Optimization) to test whether it could be applied to the tetris problem with a very simple reward function, rather than directly giving heuristics. I hoped to at the very least confirm some of the limits of this kind of algorithm on its own.

### Proximal Policy Optimization

PPO is based on the idea of trust regions, trying not to let the policy update too drastically and "fall off a cliff" into less effective policies. PPO is based on TRPO (Trust Region), which has a similar strategy but is /proximal/ meaning it is more computationally efficient, as it uses a first order rather than second order derivative to choose policy updates. Since the objective function is based on the log of probabilities and generates a negative number, it uses gradient ascent. I used the actor critic policies available for PPO rather than CNNs mainly because I currently do not have a GPU.

### Actions
The default actions provided by the environment are as follows:
NO - no_op - do nothing (move down when gravity is enabled, which it is by default)
L - move_left
R - move_right
D - move_down
CW - rotate_clockwise
CCW - rotate_counterclockwise
SWAP - swap - can swap piece to the holding area once per tetromino
DROP - hard_drop - drop the piece as far as it can fall and commit it immediately

These are the individual control actions available to a player. Using these as the action space for an RL algorithm can be problematic because it greatly increases the number of actions required for a substantial reward, so it becomes very difficult for the agent to learn a "path" to rewards. If a reward is given for staying alive, the agent can also be disincentivized to do hard drops, unnecessarily moving pieces around.

To mitigate this, actions can be grouped together into piece placements. In other words, the agent will consider the impact of each possible placement of the piece on the board (upper bound around 40 moves as 10 columns and 4 rotations) as its action space. This can create a more direct mapping between decisions the agent has to make and rewards.
The Tetris gym environment conveniently includes a wrapper for this, and I eventually decided to use it for all of my training.

### Observations
By default the environment returns a vector representation of the board, a representation of the upcoming tetrominoes, the currently held tetromino, and some other information about the state of the game.
There is also a wrapper which returns the following feature vector instead. I chose to use this because I thought the information might be more useful to the agent and it might be able to learn that these metrics matter quicker than just giving it the board itself. Making this change did improve performance.
The Feature vector observation includes:
    1. Individual column heights (10 features)
    2. Column differences (9 features)
    3. Holes count (1 feature)
    4. Maximum height (1 feature)

As discussed earlier these are basically heuristics for keeping the board low and level.

### Rewards
By default, the reward function:
When a tetromino is comitted to the board, if the game ends the reward is 0
If the game continues, the reward is 1
If any lines are cleared the reward is 1 + the score earned calculated as (rows_cleared**2) * self.width (10)

Since my goal was to test whether the agent could use a simple/direct reward function like this, I left it as is.

It is common with PPO to clip rewards to discourage making too drastic changes to the policy and "falling off the cliff" into a worse policy. I found that clipping improved performance, particularly in the early part of training. This makes sense to me, because the most important goal for the agent to learn early on is survival, rather than trying to score.

In terms of evaluation, my main metric was simply survival time. Since I never got to a point where my models would survive very long, score didn't seem that important.

## Result

In my early experiments I acheived results that were not bad compared to my other results by simply using the default parameters, no grouping of actions, etc. and just a high number of steps.
```
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 112         |
|    ep_rew_mean          | 57.3        |
| time/                   |             |
|    fps                  | 847         |
|    iterations           | 3052        |
|    time_elapsed         | 58969       |
|    total_timesteps      | 50003968    |
| train/                  |             |
|    approx_kl            | 0.051321883 |
|    clip_fraction        | 0.1         |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.266      |
|    explained_variance   | 0.926       |
|    learning_rate        | 0.0003      |
|    loss                 | 3.68        |
|    n_updates            | 30510       |
|    policy_gradient_loss | -0.00756    |
|    value_loss           | 9.5         |
-----------------------------------------
```
[video](https://drive.google.com/file/d/1j0HUdEm2a7VG8PKXsKsSI8G40tsrNpUZ/view?usp=drive_link)

After making the changes discussed above, my most promising model acheived the following result, but unfortunately at this point I ran out of time to train and couldn't see if it converged to an overall better result.
with the hyperparameters and wrappers:
Grouped actions
Feature vector representation
clip_range=0.2
learning_rate=0.00035
gamma=0.999

Note that the episode length is not directly comparable to the ungrouped actions strategy, as time is not needed for the pieces to fall.
```
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 40.8        |
|    ep_rew_mean          | 64.8        |
| time/                   |             |
|    fps                  | 623         |
|    iterations           | 62          |
|    time_elapsed         | 1629        |
|    total_timesteps      | 1015808     |
| train/                  |             |
|    approx_kl            | 0.049224086 |
|    clip_fraction        | 0.304       |
|    clip_range           | 0.2         |
|    entropy_loss         | -1.1        |
|    explained_variance   | 0.779       |
|    learning_rate        | 0.0003      |
|    loss                 | 0.138       |
|    n_updates            | 610         |
|    policy_gradient_loss | -0.0102     |
|    value_loss           | 0.424       |
-----------------------------------------
```
[video](https://drive.google.com/file/d/1ig9mPsv4Vs6T-xFD96V_Wi-xlQt4vG7q/view?usp=drive_link)

## Conclusion
None of my models acheived very good performance. 
I should have implemented a better logging suite (graphs etc.) right away, so I could see performance changing over time. Maybe I would have spent less time doing redundant training and had a better understanding.

## Ideas for further work
More experimentation with hyperparameters, including changing the reward clipping as a function of steps remaining.
I would have tried using a CNN policy but I don't have a GPU and Colab was having dependency problems I didn't have time to solve.
Would have been nice to try transfer learning
I am aware that RL algorithms can be very seed dependent, sometimes making them brittle. In other words, it may be that a model learns how to do well based on a partiuclar random seed but performance will not transfer to other random seeds. This could be a problem in future work.

## Process
Here is an extended log of the process I went through to reach my final result.

Default PPO, no wrappers
with 25,000 steps, barely does anything and loses quickly
[video](https://drive.google.com/file/d/1eTkT1WF2ii6JNdqkpI06K_F94SEq9yOH/view?usp=drive_link)
10x moar training - starts to put pieces in other places!
```
----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 72.7        |
|    ep_rew_mean          | 22.1        |
| time/                   |             |
|    fps                  | 2184        |
|    iterations           | 62          |
|    time_elapsed         | 232         |
|    total_timesteps      | 507904      |
| train/                  |             |
|    approx_kl            | 0.093055025 |
|    clip_fraction        | 0.282       |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.593      |
|    explained_variance   | 0.837       |
|    learning_rate        | 0.0003      |
|    loss                 | 0.214       |
|    n_updates            | 610         |
|    policy_gradient_loss | -0.0287     |
|    value_loss           | 1.64        |
-----------------------------------------
```
[video](https://drive.google.com/file/d/1L14Wxn6lHPVu-aUy4zInLVWyWPXiZwl-/view?usp=drive_link)

1000x and 8 parallel
```
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 110        |
|    ep_rew_mean          | 57.3       |
| time/                   |            |
|    fps                  | 846        |
|    iterations           | 3043       |
|    time_elapsed         | 58908      |
|    total_timesteps      | 49856512   |
| train/                  |            |
|    approx_kl            | 0.05443561 |
|    clip_fraction        | 0.096      |
|    clip_range           | 0.2        |
|    entropy_loss         | -0.257     |
|    explained_variance   | 0.941      |
|    learning_rate        | 0.0003     |
|    loss                 | 3.31       |
|    n_updates            | 30420      |
|    policy_gradient_loss | -0.00641   |
|    value_loss           | 8.38       |
----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 111         |
|    ep_rew_mean          | 56.6        |
| time/                   |             |
|    fps                  | 846         |
|    iterations           | 3044        |
|    time_elapsed         | 58915       |
|    total_timesteps      | 49872896    |
| train/                  |             |
|    approx_kl            | 0.055500425 |
|    clip_fraction        | 0.0991      |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.27       |
|    explained_variance   | 0.93        |
|    learning_rate        | 0.0003      |
|    loss                 | 3.34        |
|    n_updates            | 30430       |
|    policy_gradient_loss | -0.00594    |
|    value_loss           | 9.45        |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 109         |
|    ep_rew_mean          | 57.6        |
| time/                   |             |
|    fps                  | 846         |
|    iterations           | 3045        |
|    time_elapsed         | 58921       |
|    total_timesteps      | 49889280    |
| train/                  |             |
|    approx_kl            | 0.058378834 |
|    clip_fraction        | 0.104       |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.271      |
|    explained_variance   | 0.931       |
|    learning_rate        | 0.0003      |
|    loss                 | 3.65        |
|    n_updates            | 30440       |
|    policy_gradient_loss | -0.00654    |
|    value_loss           | 8.63        |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 110         |
|    ep_rew_mean          | 60.7        |
| time/                   |             |
|    fps                  | 846         |
|    iterations           | 3046        |
|    time_elapsed         | 58928       |
|    total_timesteps      | 49905664    |
| train/                  |             |
|    approx_kl            | 0.060914945 |
|    clip_fraction        | 0.107       |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.267      |
|    explained_variance   | 0.936       |
|    learning_rate        | 0.0003      |
|    loss                 | 3.48        |
|    n_updates            | 30450       |
|    policy_gradient_loss | -0.00168    |
|    value_loss           | 8.66        |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 111         |
|    ep_rew_mean          | 57.4        |
| time/                   |             |
|    fps                  | 847         |
|    iterations           | 3047        |
|    time_elapsed         | 58935       |
|    total_timesteps      | 49922048    |
| train/                  |             |
|    approx_kl            | 0.045735285 |
|    clip_fraction        | 0.101       |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.259      |
|    explained_variance   | 0.942       |
|    learning_rate        | 0.0003      |
|    loss                 | 2.35        |
|    n_updates            | 30460       |
|    policy_gradient_loss | -0.00576    |
|    value_loss           | 8.64        |
-----------------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 112        |
|    ep_rew_mean          | 58.9       |
| time/                   |            |
|    fps                  | 847        |
|    iterations           | 3048       |
|    time_elapsed         | 58942      |
|    total_timesteps      | 49938432   |
| train/                  |            |
|    approx_kl            | 0.04542879 |
|    clip_fraction        | 0.0949     |
|    clip_range           | 0.2        |
|    entropy_loss         | -0.275     |
|    explained_variance   | 0.937      |
|    learning_rate        | 0.0003     |
|    loss                 | 3.37       |
|    n_updates            | 30470      |
|    policy_gradient_loss | -0.00651   |
|    value_loss           | 8.72       |
----------------------------------------
--------------------------------------
| rollout/                |          |
|    ep_len_mean          | 115      |
|    ep_rew_mean          | 61.2     |
| time/                   |          |
|    fps                  | 847      |
|    iterations           | 3049     |
|    time_elapsed         | 58949    |
|    total_timesteps      | 49954816 |
| train/                  |          |
|    approx_kl            | 0.042204 |
|    clip_fraction        | 0.101    |
|    clip_range           | 0.2      |
|    entropy_loss         | -0.272   |
|    explained_variance   | 0.928    |
|    learning_rate        | 0.0003   |
|    loss                 | 7.04     |
|    n_updates            | 30480    |
|    policy_gradient_loss | -0.00455 |
|    value_loss           | 9.32     |
--------------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 110        |
|    ep_rew_mean          | 59.3       |
| time/                   |            |
|    fps                  | 847        |
|    iterations           | 3050       |
|    time_elapsed         | 58956      |
|    total_timesteps      | 49971200   |
| train/                  |            |
|    approx_kl            | 0.06446559 |
|    clip_fraction        | 0.112      |
|    clip_range           | 0.2        |
|    entropy_loss         | -0.273     |
|    explained_variance   | 0.931      |
|    learning_rate        | 0.0003     |
|    loss                 | 3.41       |
|    n_updates            | 30490      |
|    policy_gradient_loss | -0.00629   |
|    value_loss           | 9.25       |
----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 111         |
|    ep_rew_mean          | 58.4        |
| time/                   |             |
|    fps                  | 847         |
|    iterations           | 3051        |
|    time_elapsed         | 58962       |
|    total_timesteps      | 49987584    |
| train/                  |             |
|    approx_kl            | 0.048811696 |
|    clip_fraction        | 0.1         |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.261      |
|    explained_variance   | 0.932       |
|    learning_rate        | 0.0003      |
|    loss                 | 3.96        |
|    n_updates            | 30500       |
|    policy_gradient_loss | -0.00602    |
|    value_loss           | 9.53        |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 112         |
|    ep_rew_mean          | 57.3        |
| time/                   |             |
|    fps                  | 847         |
|    iterations           | 3052        |
|    time_elapsed         | 58969       |
|    total_timesteps      | 50003968    |
| train/                  |             |
|    approx_kl            | 0.051321883 |
|    clip_fraction        | 0.1         |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.266      |
|    explained_variance   | 0.926       |
|    learning_rate        | 0.0003      |
|    loss                 | 3.68        |
|    n_updates            | 30510       |
|    policy_gradient_loss | -0.00756    |
|    value_loss           | 9.5         |
-----------------------------------------
```
Mean reward: 61.498 +/- 18.07
Observations:
- not rotating very much, but does sometimes
- actually clears some lines now! but seemingly by chance
- seems to "understand" that pieces should be distributed evenly throughout the board to live longer, but not intricacies of shapes fitting together
[video](https://drive.google.com/file/d/1j0HUdEm2a7VG8PKXsKsSI8G40tsrNpUZ/view?usp=drive_link)

subprocvecenv sped things up

Clipping the reward seemed to make a big difference. after only 1M steps:
```
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 82.1        |
|    ep_rew_mean          | 26.9        |
| time/                   |             |
|    fps                  | 2818        |
|    iterations           | 62          |
|    time_elapsed         | 360         |
|    total_timesteps      | 1015808     |
| train/                  |             |
|    approx_kl            | 0.059507556 |
|    clip_fraction        | 0.156       |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.268      |
|    explained_variance   | 0.968       |
|    learning_rate        | 0.0003      |
|    loss                 | 0.0763      |
|    n_updates            | 610         |
|    policy_gradient_loss | -0.011      |
|    value_loss           | 0.328       |
-----------------------------------------
```
[video](https://drive.google.com/file/d/1_T7YYFgEa86nh32Bt7zl0lzYk2npmyUB/view?usp=drive_link)

More training with clipped reward has diminishing returns quickly and actually doesn't surpass the default parameters 50M steps.
It seems like the model learned how to survive faster, but may not have had an incentive to score beyond that.
Note that the clipping can also be done in the model parameter itself.

Now trying Grouping actions (no clipping), had to change to MlpPolicy because the observation space is different.
It makes more sense to group the actions together into a set of possible "moves," places the active tetromino can go on the board, than to learn individual movement actions.
I'm pretty sure this makes the episodes go faster, so the lengths are not directly comparable.
```
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 24.6       |
|    ep_rew_mean          | 24.8       |
| time/                   |            |
|    fps                  | 385        |
|    iterations           | 62         |
|    time_elapsed         | 2636       |
|    total_timesteps      | 1015808    |
| train/                  |            |
|    approx_kl            | 0.15808865 |
|    clip_fraction        | 0.126      |
|    clip_range           | 0.2        |
|    entropy_loss         | -1.42      |
|    explained_variance   | 0.791      |
|    learning_rate        | 0.0003     |
|    loss                 | 5.14       |
|    n_updates            | 610        |
|    policy_gradient_loss | 0.0209     |
|    value_loss           | 10.4       |
----------------------------------------
```
Observations:
- many more rotations are happening
- model never chooses to store a piece, this is fine with me for now
[video](https://drive.google.com/file/d/1d6lCcRtRlNc8YQfeueD-RBuratok81L-/view?usp=drive_link)

Now trying the wrapper that changes the observation space to a feature vector related to how good the board is.
```
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 30.2        |
|    ep_rew_mean          | 31.2        |
| time/                   |             |
|    fps                  | 571         |
|    iterations           | 31          |
|    time_elapsed         | 889         |
|    total_timesteps      | 507904      |
| train/                  |             |
|    approx_kl            | 0.052517056 |
|    clip_fraction        | 0.306       |
|    clip_range           | 0.2         |
|    entropy_loss         | -2.1        |
|    explained_variance   | 0.877       |
|    learning_rate        | 0.0003      |
|    loss                 | 3.73        |
|    n_updates            | 300         |
|    policy_gradient_loss | 0.0203      |
|    value_loss           | 8.25        |
-----------------------------------------
```
This surpassed the previous model even with only 500k steps.

Adding the clipping back in... made it slightly worse this time, so maybe the grouped actions are doing a better job and it isn't needed.
Removing the clipping and training for 5M steps:
```
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 29.8       |
|    ep_rew_mean          | 28.8       |
| time/                   |            |
|    fps                  | 591        |
|    iterations           | 31         |
|    time_elapsed         | 858        |
|    total_timesteps      | 507904     |
| train/                  |            |
|    approx_kl            | 0.06969376 |
|    clip_fraction        | 0.302      |
|    clip_range           | 0.2        |
|    entropy_loss         | -2.12      |
|    explained_variance   | 0.962      |
|    learning_rate        | 0.0003     |
|    loss                 | 0.744      |
|    n_updates            | 300        |
|    policy_gradient_loss | 0.0254     |
|    value_loss           | 2.01       |
----------------------------------------
```
Led to a very strange way of playing tetris, filling columns to delay the game.
[video](https://drive.google.com/file/d/1kzvr5CIAXbPINHZMLz0MYYxsM1fKNPvx/view?usp=drive_link)

With clipping and hyperparameter adjustment
```
---------------------------------------
| rollout/                |           |
|    ep_len_mean          | 40.5      |
|    ep_rew_mean          | 60.8      |
| time/                   |           |
|    fps                  | 574       |
|    iterations           | 62        |
|    time_elapsed         | 1767      |
|    total_timesteps      | 1015808   |
| train/                  |           |
|    approx_kl            | 0.0456689 |
|    clip_fraction        | 0.279     |
|    clip_range           | 0.2       |
|    entropy_loss         | -0.991    |
|    explained_variance   | 0.782     |
|    learning_rate        | 0.0003    |
|    loss                 | 0.139     |
|    n_updates            | 610       |
|    policy_gradient_loss | -0.00936  |
|    value_loss           | 0.285     |
---------------------------------------
```
[video](https://drive.google.com/file/d/1ig9mPsv4Vs6T-xFD96V_Wi-xlQt4vG7q/view?usp=drive_link)

[All videos](https://drive.google.com/drive/folders/1_2x5jwBlrk3gNLlKrzP5NjJsxzYeD-sf?usp=drive_link)
- Citations

Machado et al. 2017. [Revisiting the Arcade Learning Environment : Evaluation Protocols and Open Problems for General Agents](https://arxiv.org/pdf/1709.06009)
Matt Stevens and Sabeek Pradhan. 2016. [Playing Tetris with Deep Reinforcement Learning](https://cs231n.stanford.edu/reports/2016/pdfs/121_Report.pdf)
Maximilian Weichart and Philipp Hartl. July 20, 2024. [Piece by Piece: Assembling a Modular Reinforcement Learning Environment for Tetris.](https://easychair.org/publications/preprint/154Q)
  [Github for tetris env](https://github.com/Max-We/Tetris-Gymnasium)
[ozhi - tetris-ai (minimax).](https://github.com/ozhi/tetris-ai/tree/master)
Schulman et al. 2017. [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347)
