#!/usr/bin/env bash

container_id=`sudo docker ps | grep capstone | awk '{print $1;}'`
sudo docker exec -it $container_id /bin/bash


