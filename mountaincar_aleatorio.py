#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 13:48:48 2020

@author: uli
"""

import gym # Se importa la libreria gym
entorno =gym.make("MountainCar-v0") #Se define el entorno del videojuego a ussar, en este caso es MountainCar
partidas = 100 #el numero total de episodios o intentos


for episodio in range(partidas): #realizara las partidas o episodios
    done = False #si no termina
    observacion = entorno.reset() #el agente inicia en el entorno, abre los ojos 
    recompenza_total =0.0 #variable de la recompenza total
    paso =0 #variable de los pasos 
    while not done: #mientras no termine, se ejecutara lo siguiente
        entorno.render() #se muestra en pantalla el juego
        action=entorno.action_space.sample() #elije una accion aleatoria
        next_state, reward, done, info = entorno.step(action) 
        recompenza_total += reward #la recompenza total del agente
        paso += 1  #pasa al siguiente paso
        observacion = next_state #pasa al siguiente estado
        

    print("\n Episodio número {} finalizado con {} iteraciones. Recompensa final={}".format(episodio, paso+1, recompenza_total))    
      
entorno.close() #se cierra el entorno
    