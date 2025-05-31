# neural-net
This project features implementation of a simple feedforward neural network combined with an evolutionary algorithm for optimization, written in Java. In the genetic algorithm, an individual corresponds to a flattened array of the neural network's parameters.
## Features
* The neural network architecture is fully customizable: user defines the number of layers and activation functions per layer
* Roulette wheel selection is used to probabilistically select individuals based on their fitness
* Gaussian mutation is applied to chromosome elements, with tunable mutation probability and standard deviation
## How to run
This is an example of CLI arguments passed to application:
```bash
--train sine_train.txt --test sine_test.txt --nn 5s-5s --popsize 10 --elitism 1 --p 0.1 --K 0.1 --iter 10000
```
Following arguments need to be provided:
* train/test - path to train/test data
* nn - architecture of neural net, in the format: (number of layers)(activation function) for each layer, separated with "-"
* popsize - population size
* elitism - number of best individuals to retain
* p - probability of mutation
* K - stdev of mutation
* iter - number of iterations

Available activation functions are:
* s - sigmoid
* r - ReLU
* l - LReLU
* t - tanh
