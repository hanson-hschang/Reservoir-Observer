# Reservoir-Observer

## Introduction 
This repository gives two examples with reservoir observer and it is based on this paper: [Reservoir observers: Model-free inference of unmeasured variables in chaotic systems](https://aip.scitation.org/doi/10.1063/1.4979665) The two dynamic systems chosen are [Rössler](https://en.wikipedia.org/wiki/R%C3%B6ssler_attractor) and [Lorenz](https://en.wikipedia.org/wiki/Lorenz_system).



## Download
1. Move to your prefered directory.
```
$ cd dir
```

Note: Replace `dir` with your actual prefered directory path.

2. Clone this repository, which will create a new folder, with the name: Reservoir-Observer, under your prefered directory.
```
$ git clone https://github.com/hanson-hschang/Reservoir-Observer.git
```

3. Run Rössler example.
```
$ python ReservoirObserver.py rossler
```
![Rössler example](https://github.com/hanson-hschang/Reservoir-Observer/blob/master/rossler.png)


4. Run Lorenz example.
```
$ python ReservoirObserver.py lorenz
```
![Lorenz example](https://github.com/hanson-hschang/Reservoir-Observer/blob/master/lorenz.png)
