### Logic gates with Neural Nets evolved with Genetic Algorithms

<center>
  ![meme](./aux/logic-gates-with-neural-nets.jpg)
</center>

#### Evolution of the neural net
<center>
  ![gif-net-evolve.gif](./aux/gif-net-evolve.gif)
</center>

The network has 2 inputs (I), 1 bias (B), 1 output (O) with sigmoid, and only one hidden neuron with Gaussian activation. This network solves the XOR logic gate.

#### Fitness
<center>
  ![fitness](./aux/fitness.png)
</center>

#### Technical notes
To continuously update the network graph, I use sockets to communicate with my Python program (which draws the graph).
