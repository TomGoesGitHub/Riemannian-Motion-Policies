# Dynamic Obstacle Avoidance for Learning Robots with Riemannian Motion Policies

## Abstract
Riemannian Motion Policies (RMPs) represent a motion-control framework in robotics, leveraging the geometric principles of Riemannian manifolds to design and execute motion trajectories. Traditional motion
planning approaches often operate within Euclidean spaces, limiting the adaptability and efficiency of the resulting policies in complex, high-dimensional environments. RMPs overcome these limitations by
embedding the motion planning problem in a Riemannian manifold, where the curvature and intrinsic properties of the space are naturally incorporated into the motion dynamics. This enables the generation of
smooth, adaptive, and collision-free trajectories that can navigate complex environments and handle dynamic obstacles more effectively.

## Demo
(GIF may take a few seconds to load)

![06_cluttered_environment-ezgif com-optimize](https://github.com/TomGoesGitHub/Riemannian-Motion-Policies/assets/81027049/e7f751ab-6d69-4a0e-aeff-07e03d7c8f9a)

## Architecture
Several iterations of the RMP framework have been presented in the literature. The original implementation [Ratliff, 2018] treats each map individually, which makes it computationally inefficient as it does not consider
the structure of the kinematic chain of the robot, which leads to a lot of redundant computations (e.g. the map to the robot hand implicitly contains the map to the wrist). Therefore, in the next iteration [Cheng, 2020],
the kinematic chain is modelled as the so-called RMPgraph, which arranges different taskmaps in a graph structure, which allows a forward/backward-pass-style update scheme, that exploits the chain rule for jacobians.
However, the implementation of the RMPgraph is not trivial and jacobians still must be defined manually, which makes the implementation tedious. Therefore, in the RMP2 framework [Li, 2021] auto differentiation is used to
generate the graph and the jacobians automatically, such that only the forward pass must be implemented.

In this project an RMP2 inspired version of the RMP framework was implemented from scratch, using the autodifferentiation library tensorflow. Pybullet was used for physics simulation. 
The Figure below shows the overall architecture of my approach schematically.

![architecture](https://github.com/TomGoesGitHub/Riemannian-Motion-Policies/assets/81027049/1e73c782-0876-4bc2-b446-da32e1f5dbcc)

## Modular Policy Combination
RMPs provide a straightforward method for combining multiple motion policies and transforming such policies from one space (such as the task space) to another (such as the configuration space) in geometrically consistent ways.
The RMP framework enables the fusion of motion policies from different motion generation paradigms, such as dynamical systems, dynamic movement primitives or optimal control, thus unifying disparate techniques from the literature.

The table below shows RMPs for specific policies in action. (GIFs may take a few seconds to load)
| Policy | 2D | 3D |
| ------------- | ------------- |------------- |
| Target Reaching  | ![01_target_rmp_only](https://github.com/TomGoesGitHub/Riemannian-Motion-Policies/assets/81027049/130db396-2db0-4c25-aa24-e9211c6f4ed5) | ![01_target_rmp_only](https://github.com/TomGoesGitHub/Riemannian-Motion-Policies/assets/81027049/08605985-9a56-4486-965e-14839039fb3c)|
| Redundancy Resolution / Nullspace Control | ![02_jointspace_biasing](https://github.com/TomGoesGitHub/Riemannian-Motion-Policies/assets/81027049/79b53348-2cd0-42b8-b698-5361962d90fb) |  ![04_nullspace_control](https://github.com/TomGoesGitHub/Riemannian-Motion-Policies/assets/81027049/d068d90d-9819-4a90-919b-0748c3767d0c) |
| Obstacle Avoidance  | ![05_obstacle_avoidance](https://github.com/TomGoesGitHub/Riemannian-Motion-Policies/assets/81027049/000977ba-d46c-4e0b-bc11-e8ace8b4c8c3) | ![05_obstacle_avoidance](https://github.com/TomGoesGitHub/Riemannian-Motion-Policies/assets/81027049/8b971644-3922-454a-8ac7-fac2c40501e1) |

## Value for Robot Learning
Therefore, RMPs offer great potential for applications in the field of robot learning as well. Changing environments pose a challenge for classical agents. Learned policies may become useless when being confronted with variable
obstacles if the interaction has not been sufficiently learned during training. RMPs offer the possibility to react dynamically to these obstacles by combining the potential fields of the obstacles with the agent's policy. Thus,
the primary task can be trained without obstacles (which significantly simplifies the training process) and the obstacle avoidance can be integrated afterwards with the help of the RMP framework.
