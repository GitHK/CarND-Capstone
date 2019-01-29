#!/usr/bin/env bash

sudo docker run -p 4567:4567 -v $PWD:/capstone:delegated --rm -it capstone
