#!/bin/bash
while true
do
	python3.11 -c "import data_generator;data_generator.run()"
	echo "collect.sh: press ctrl+c to stop..."
	sleep 10
done
