#!/bin/bash

for fiter in data/*; do
	if ["$fiter" == "data/MNIST_k10_clustered_notuniform.simulated"] ; then 
		continue;
	fi
	for kiter in 1 2 5 10; do
		python src/main.py --input "${fiter}" --k "${kiter}"
	done
done
