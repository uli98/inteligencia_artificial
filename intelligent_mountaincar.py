#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 12:36:56 2020

@author: uli
"""

import gym
import numpy as np


# EPISILON_MIN : vamos aprendiendo, mientras el incremento de aprendizaje sea superior a dicho valor
# episodios_totales : número máximo de iteraciones que estamos dispuestos a realizar
# pasos_episodio: número máximo de pasos a realizar en cada episodio
# ALPHA: radio de aprendizaje del agente 
# GAMMA: factor de descuento del agente (Lo que se pierde de un paso a otro para insentivar al agente a llegar mas rapido)
# NUM_DIvisiones: número de divisiones en el caso de discretizar el espacio de estados continuo.  

episodios_totales = 50000 
pasos_episodio = 200 
epsilon_min = 0.005
numero_maximo_pasos = episodios_totales * pasos_episodio #el numero total de pasos que se dara en el algoritmo sera el producto de los episodoios totales y los pasos por episodio
epsilon_decremento = 500 * epsilon_min / numero_maximo_pasos # el valor que se perdera cada vez que se aprende 
ALPHA = 0.05
GAMMA = 0.98
num_div = 30

class MountainCar(object): #Se define la clase
    
    def __init__(self, entorno): #Se define la funcion o metodo 
         self.tam_obs = entorno.observation_space.shape
         self.alto_obs =entorno.observation_space.high
         self.bajo_obs =entorno.observation_space.low
         self.obs_div= num_div
         self.ancho_obs= (self.alto_obs-self.bajo_obs)/self.obs_div
         
         self.tam_accion= entorno.action_space.n
         self.q =np.zeros((self.obs_div+1, self.obs_div+1, self.tam_accion)) 
         self.alpha = ALPHA
         self.gamma= GAMMA
         self.epsilon= 1.0
         
    def discretize(self, obs):
        
       return tuple(((obs-self.bajo_obs)/self.ancho_obs).astype(int))
    
    def get_action(self, obs):
        
         discrete_obs =self.discretize(obs)
         if self.epsilon > epsilon_min:
                self.epsilon -= epsilon_decremento
         if np.random.random() > self.epsilon:
            return np.argmax(self.q[discrete_obs])
         else:
            return np.random.choice([a for a in range(self.tam_accion)])
        
    def learn(self, obs, action, reward, next_obs):
             
        discrete_obs =self.discretize(obs)
        discrete_next_obs = self.discretize(next_obs)
        self.q[discrete_obs][action] += self.alpha*(reward + self.gamma * np.max(self.q[discrete_next_obs]) - self.q[discrete_obs][action])
        
        
        
def entrenar(agente, entorno):
    mejor_recompenza = -float('inf')
    for episodio in range(episodios_totales):
        done = False
        obs = entorno.reset()
        recompenza_total = 0.0
        while not done:
            action = agente.get_action(obs)# Acción elegida según la ecuación de Q-LEarning
            next_obs, reward, done, info = entorno.step(action)
            agente.learn(obs, action, reward, next_obs)
            obs = next_obs
            recompenza_total += reward
        if recompenza_total > mejor_recompenza:
            mejor_recompenza = recompenza_total
        print("EPisodio número {} con recompensa: {}, mejor recompensa: {}, epsilon: {}".format(episodio, recompenza_total, mejor_recompenza, agente.epsilon))
    
    ## De todas las políticas de entrenamiento que hemos obtenido devolvemos la mejor de todas
    return np.argmax(agente.q, axis = 2)
    
def test(agente, entorno, politica):
       done = False
       obs = entorno.reset()
       recompenza_total = 0.0
       while not done:
           action = politica[agente.discretize(obs)] #acción que dictamina la política que hemos entrenado
           next_obs, reward, done, info = entorno.step(action)
           obs = next_obs
           recompenza_total += reward
       return recompenza_total
   

if __name__ == "__main__":
    entorno = gym.make("MountainCar-v0")
    agente = MountainCar(entorno)
    politica_aprendizaje = entrenar(agente, entorno)
    monitor_path = "./monitor_output"
    entorno = gym.wrappers.Monitor(entorno, monitor_path, force = True)
    for _ in range(1000):
        test(agente, entorno, politica_aprendizaje)
    entorno.close()