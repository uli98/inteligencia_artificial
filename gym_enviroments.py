#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 12:44:59 2020

@author: uli
"""

from gym import envs 
env_names = [env.id for env in envs.registry.all()]
for name in sorted(env_names) :
    print(name)
    