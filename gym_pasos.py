#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 12:06:34 2020

@author: uli
"""


import gym 

environment = gym.make("Qbert-v0") 
episodios = 10 #son las veces que se juega la partida
iteraciones = 500 #son los pasos o iteraciones en los que se jugara cada partida

for episode in range(episodios): # es un bucle en el cual se puede entender que el agente llega a un entorno y lo pasara en este caso 10 veces
    observacion = environment.reset() # Aqui el agente abre los ojos y ve el entorno
    for step in range(iteraciones): #Numero de veces que pasara por ese entorno el agente
        environment.render() # se observa en pantalla
        action = environment.action_space.sample()## Tomamos una decisión aleatoria...
        next_state, reward, done, info = environment.step(action) #algoritmo bellman 
        observacion = next_state
        
        if done is True: # el agente sabe cuando termina un episodio 
            print("\n Episodio #{} terminado en {} pasos.".format(episode, step+1))
            break
        
environment.close() # Cerramos la sesión de Open AI Gym
