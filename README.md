# Cross-Tool and Cross-Behavior Perceptual Knowledge Transfer for Grounded Object Recognition

**Abstract:**

> Humans learn about objects via interaction and using multiple perceptions, such as vision, sound, and touch.
While vision can provide information about an object's appearance, non-visual sensors, such as audio and haptics, can provide information about its intrinsic properties, such as weight, temperature, hardness, and the object's sound.
Using tools to interact with objects can reveal additional object properties that are otherwise hidden (e.g., knives and spoons can be used to examine the properties of food, including its texture and consistency.)
Robots can use tools to interact with objects and gather information about their implicit properties via non-visual sensors.
However, a robot's model for recognizing objects using a tool-mediated behavior does not generalize to a new tool or behavior due to differing observed data distributions.
To address this challenge, we propose a framework to enable robots to transfer implicit knowledge about granular objects across different tools and behaviors.
The proposed approach learns a shared latent space from multiple robots' contexts produced by respective sensory data while interacting with objects using tools.
We collected a dataset using a UR5 robot that performed 5,400 interactions using 6 tools and 6 behaviors on 15 granular objects and tested our method on cross-tool and cross-behavioral transfer tasks.
Our results show the less experienced target robot can benefit from the experience gained from the source robot and perform recognition on a set of novel objects.
We have released the code, datasets, and additional results: https://github.com/gtatiya/Tool-Knowledge-Transfer.

## Installation

`Python 3.10` and `MATLAB R2022b` are used for development.

```
pip install -r requirements.txt
```

### MATLAB

[Install MATLAB Engine API for Python](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html) <br>
MATLAB Dependencies: Statistics and Machine Learning Toolbox

## Dataset

- [Download the dataset](https://tufts.box.com/s/e22rlxx3tatmgjvkp1e4xdo10jnhb79v)
- Dataset details can be found on the [dataset webpage](https://www.eecs.tufts.edu/~gtatiya/pages/2023/UR5_Tool_Dataset.html).

### Discretized Representation

- [Visualization of discretized modalities](DatasetVisualization.ipynb) <br>
- <img src="figs/Features.jpg" alt="drawing" width="600px"/>

### Dataset Collection

<img src="figs/Robot_and_Sensors.jpg" alt="drawing" width="600px"/>

UR5: https://youtu.be/0-tLtI16X2U <br>

<img src="figs/UR5_Tool_Dataset_Demo.gif" alt="drawing" width="350" height="250"/>

## How to run the code?

### Transfer robot knowledge

```
python transfer_robot_knowledge.py -across <TRANSFER TASK> -augment-trials <NO. OF TRIALS> -classifier-name <CLASSIFIER> -num-folds <NO. OF FOLDS> -increment-train-objects
```
Example:
```
python transfer_robot_knowledge.py -across tools -augment-trials 10 -classifier-name SVM-RBF -num-folds 10 -increment-train-objects
```

## Results

### Illustrative Example

<img src="figs/IE.jpg" alt="drawing" width="900px"/>

### Accuracy Results of Object Recognition

#### Cross-tool sensorimotor transfer

<img src="figs/Cross-tool_Top_5_Minimum_Accuracy_Delta.png" alt="drawing" width="900px"/>

#### Cross-behavioral sensorimotor transfer

<img src="figs/Cross-behavioral_Top_5_Minimum_Accuracy_Delta.png" alt="drawing" width="900px"/>

### Accuracy Delta Results

#### Cross-tool sensorimotor transfer

<table>
<tbody>
<tr>
<td><img src="figs/Projections_Stirring-Slow_to_Stirring-Slow.png" alt="drawing" width="350" height="180"/></td>
<td><img src="figs/Projections_Stirring-Fast_to_Stirring-Fast.png" alt="drawing" width="350" height="180"/></td>
</tr>
<tr>
<td><img src="figs/Projections_Stirring-Twist_to_Stirring-Twist.png" alt="drawing" width="350" height="180"/></td>
<td><img src="figs/Projections_Whisk_to_Whisk.png" alt="drawing" width="350" height="180"/></td>
</tr>
<tr>
<td><img src="figs/Projections_Poke_to_Poke.png" alt="drawing" width="350" height="180"/></td>
<td></td>
</tr>
</tbody>
</table>

#### Cross-behavioral sensorimotor transfer

<table>
<tbody>
<tr>
<td><img src="figs/Projections_Metal-Scissor_to_Metal-Scissor.png" alt="drawing" width="350" height="180"/></td>
<td><img src="figs/Projections_Metal-Whisk_to_Metal-Whisk.png" alt="drawing" width="350" height="180"/></td>
</tr>
<tr>
<td><img src="figs/Projections_Plastic-Knife_to_Plastic-Knife.png" alt="drawing" width="350" height="180"/></td>
<td><img src="figs/Projections_Plastic-Spoon_to_Plastic-Spoon.png" alt="drawing" width="350" height="180"/></td>
</tr>
<tr>
<td><img src="figs/Projections_Wooden-Chopstick_to_Wooden-Chopstick.png" alt="drawing" width="350" height="180"/></td>
<td><img src="figs/Projections_Wooden-Fork_to_Wooden-Fork.png" alt="drawing" width="350" height="180"/></td>
</tr>
</tbody>
</table>

### Tools and Behaviors Transfer Relationships

<img src="figs/Tools_behaviors_2D_neighborhood_graph.png" alt="drawing" width="350" height="250"/>
