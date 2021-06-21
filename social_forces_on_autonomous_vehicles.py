# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 16:20:19 2021

@author: Fauez
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio

############## INPUTS AND PARAMETERS ###########################################

### Choice of scenario ---!
Scenario = 7
### Integration step ---!
Delta_t = 0.01

### SOCIAL FORCES ------------------------------------------------------------

### Scale coeficient for Target Social Force
tau = 0.6
### Scale coeficient for Repulsion Force
beta_1 = 3
### Sensibility coeficient for Repulsion Force
beta_2 = 3
### Agent prevision for next position
Delta_t_prevision = 2

### VISUALIZATION ------------------------------------------------------------
### Grid of steps in simulation
a = 3 #x
b = 2 #y

############## OBJECTS ########################################################

######## AGENT OBJECT -------------------------------------------------------

class agent:
    
    def __init__(self, uid, position, velocity, target = True, active = True, desired_speed = 5):
        self.uid = uid
        self.target = target
        self.active = active
        self.position = [np.array(position)]
        self.velocity = [np.array(velocity)]
        self.resultant = []
        self.desired_speed = desired_speed
        self.desired_areas = []
        self.desired_area = []
        
        #These terms will help to implementate the Runge Kutta method
        self.position_float = np.array(position)
        self.velocity_float = np.array(velocity)
        self.kx = []
        self.kv = []
        
    def check_agent_in_area(self):
        r_agent = self.position[-1]
        r_area = self.desired_area.position
        
        if vector_norm(r_agent - r_area) <= self.desired_area.radius:
            return True
        else:
            return False
        
    def desired_direction(self):
        desired_areas = self.desired_areas
        if self.check_agent_in_area():
            if self.desired_area.last == True :
                self.active = False
            else:    
                index = self.desired_areas.index(self.desired_area)
                self.desired_area = desired_areas[index+1]
            
        desired_area = self.desired_area
        e = desired_area.position - self.position[-1]
        return get_direction(e)
        
    
######### BORDER OBJECT -------------------------------------------------------

class border: 
    def __init__(self, uid, P_start, P_end):
        self.uid = uid
        self.P_start = np.array(P_start)
        self.P_end = np.array(P_end)
        self.Qx = np.array([P_start[0], P_end[0]])
        self.Qy = np.array([P_start[1], P_end[1]])
        
    def dist_border(self, P): 
        d = self.P_end - self.P_start
        alpha = np.dot( P - self.P_start , d) / np.dot( d , d )
        Q = self.P_start + alpha * d
        return Q, alpha
        
######## DESIRED AREA OBJECT -------------------------------------------------
    
class desired_area:
    def __init__(self, uid, P, r=3, last = False):
        self.position = np.array(P)
        self.radius = r
        self.uid = uid
        self.last = last
    
    def get_position(self):
        return self.position
    
    def get_radius(self):
        return self.radius
    
    def get_area(self):
        r = self.radius
        area = np.pi * r**2
        return area
    


#### SCENARIO DEFINITION -----------------------------------------------------

def Scenarios(num):
    agents = []
    borders = []
    desired_areas = []
    xlim = (0,0)
    ylim = (0,0)
    t_end = 100
    
    if num == 1:
        'test the avoidance Force in a single obstacle and just one agent'
        agent0 = agent(0, [0 , 0], [10, 1], target = False)
        agents = [agent0]
        
        border0 = border(0, [-50, 30], [800,30])
        border1 = border(1, [-50, -30], [800, -30])
        borders = [border0, border1]
        
        xlim=(-50, 800)
        ylim=(-40, 40)   
        t_end = 60
              
    if num == 2:
        'ten agents in a road'
        border0 = border(0, [-50, 30], [1500,30])
        border1 = border(1, [-50, -30], [1500, -30])
        borders = [border0, border1]
        agents = []
        
        for i in range(10):
            if i%2 == 0:
                j = -1
            else:
                j = 1
                
            agents.append( agent(i, [0, -25+(50/10)*i], [5,j*0.3], target = False ))
        t_end = 200
        xlim=(-50, 800)
        ylim=(-40, 40)
    
    if num == 3:
        border0 = border(0, [-50, 30], [1500,30])
        border1 = border(1, [-50, -30], [1500, -30])
        borders = [border0, border1]
        
        agent0 = agent(0, [0 , 0.01], [10, 0], target = False)
        agent1 = agent(1, [1400 , -0.01], [-10, 0], target = False)
        agents = [agent0, agent1]
        
        t_end = 200
        xlim=(-50, 800)
        ylim=(-40, 40)
    
    if num == 4:
        'first test of target force'
        area1 = desired_area(0, [50,0], 5)
        area2 = desired_area(1, [50, -50], 5, last = True)
        desired_areas = [area1, area2]
        
        agent0 = agent(0, [0,0], [0,0])
        agent0.desired_areas = desired_areas
        agent0.desired_area = area1
        agents = [agent0]
        
        border0 = border(0, [0,5], [55,5])
        border1 = border(1, [0,-5], [45,-5])
        border2 = border(2, [55,5], [55, -50])
        border3 = border(3, [45,-5], [45,-50])
        borders = [border0, border1, border2, border3]
        
        t_end = 20
    
    if num == 5:
        'test the situation of overtaking in a road'
        border0 = border(0, [-50,0],[20,0])
        border1 = border(1, [-50, 5],[20,5])
        borders = [border0, border1]
        
        agent0 = agent(0, [-40, 1.25], [5,0.5], desired_speed=15)
        area0 = desired_area(0, [20, agent0.position[0][1]], 2, last = True)
        agent0.desired_area = area0 
        agent0.desired_areas = [area0]
           
        agent1 = agent(1,[-30,1.25], [5,0])
        area1 = desired_area(1, [20, agent1.position[0][1]], 2, last = True)
        agent1.desired_area = area1 
        agent1.desired_areas = [area1]
        
        agent2 = agent(2, [-40,3.75],[5,0])
        area2 = desired_area(2, [20, agent2.position[0][1]], 2, last = True)
        agent2.desired_area = area2
        agent2.desired_areas = [area2]
        
        agents = [agent0, agent1, agent2]
        desired_areas = [area0, area1, area2]
        
        ylim = (-5,10)
        xlim = (-50,25)
        
        t_end = 9
    
    if num == 6:
        'two way street'
        border0 = border(0, [-50,0],[20,0])
        border1 = border(1, [-50, 12],[20,12])
        borders = [border0, border1]
        
        # agent0 = agent(0, [-50, 1.5], [5,0])
        # area0 = desired_area(0, [border0.P_end[0], agent0.position[0][1]], last = True)
        # agent0.desired_area = area0
        
        # agent1 = agent(1, [-40, 4.5], [5,0])
        # area1 = desired_area(0, [border0.P_end[0], agent1.position[0][1]], last = True)
        # agent1.desired_area = area1
        
        # agent2 = agent(2, [-20, 4.5], [5,0])
        # area2 = desired_area(2, [border0.P_end[0], agent2.position[0][1]], last = True)
        # agent2.desired_area = area2
        
        # agent3 = agent(3, [-15, 1.5], [5,0])
        # area3 = desired_area(3, [border0.P_end[0], agent3.position[0][1]], last = True)
        # agent3.desired_area = area3
        
        # agent4 = agent(4, [20, 10.5], [-5,0])
        # area4 = desired_area(4, [border0.P_start[0], agent4.position[0][1]], last = True)
        # agent4.desired_area = area4
        
        # agent5 = agent(5, [5, 10.5], [-5,0])
        # area5 = desired_area(5, [border0.P_start[0], agent5.position[0][1]], last = True)
        # agent5.desired_area = area5
        
        # agent6 = agent(6, [15, 7.5], [-5,0])
        # area6 = desired_area(6, [border0.P_start[0], agent6.position[0][1]], last = True)
        # agent6.desired_area = area6
        
        agent7 = agent(7, [10,7.5],[-10,0])
        area70 = desired_area(70, [-20,1.5])
        area71 = desired_area(71, [border0.P_start[0], agent7.position[0][1]], last = True)
        agent7.desired_area = area70
        agent7.desired_areas = [area70, area71]
        
        n = 5
        agents_left = [0]*n
        x_position_left = [-50,-45,-40,-35,-30,-25,-20]
        y_position_left = [1.5,4.5]
        for i in range(n):
            x = np.random.choice(x_position_left)
            x_position_left.pop(x_position_left.index(x))
            y = np.random.choice(y_position_left)
            vx = 5+ np.random.rand()*5
            agents_left[i] = agent(i, [x,y], [vx, 0], desired_speed = vx )
            area = desired_area(i, [20,agents_left[i].position[0][1] ], last = True)
            agents_left[i].desired_area = area
            agents_left[i].desired_areas = [area]
            
        agents_right = [0]*n
        x_position_right = [20,15,5,0,-5,-10]
        y_position_right = [7.5,10.5]
        for i in range(n):
            x = np.random.choice(x_position_right)
            x_position_right.pop(x_position_right.index(x))
            y = np.random.choice(y_position_right)
            vx = -(5+ np.random.rand()*5)
            agents_right[i] = agent(i+n, [x,y], [vx, 0], desired_speed = abs(vx) )
            area = desired_area(i, [-50,agents_right[i].position[0][1] ], last = True)
            agents_right[i].desired_area = area
            agents_right[i].desired_areas = [area]
            
            
        for i in range(n):
            agents.append(agents_left[i])
            agents.append(agents_right[i])
        agents.append(agent7)
        xlim = tuple(border0.Qx)
        ylim = ( border0.Qy[0]-5, border1.Qy[0] +5 )
        t_end = 13
        
    if num == 7:
        'Crossroads'
        border0 = border(0, [0,30], [30,30])
        border1 = border(1, [0,42], [30,42])       
        border2 = border(2, [30, 0], [30,30])
        border3 = border(3, [42,0], [42,30])   
        border4 = border(4, [30,42], [30,72])
        border5 = border(5, [42,42], [42,72])
        border6 = border(6, [42,30], [72, 30])
        border7 = border(7, [42,42], [72,42])
        
        borders = [border0, border1, border2, border3, border4, border5, border6,
                   border7]
        
        agent0 = agent(0, [20,31.5], [5,0])
        area00 = desired_area(0, [36,agent0.position[0][1]])
        area01 = desired_area(1, [40.5, 72], 3 , last = True)
        agent0.desired_area = area00
        agent0.desired_areas=[area00, area01]
        
        agent1 = agent(1, [0,31.5], [5,0])
        area10 = area00
        area11 = desired_area(2, [72, agent1.position[0][1]], last =  True)
        agent1.desired_area = area10
        agent1.desired_areas = [area10, area11]
        
        agent2 = agent(2, [10,31.5], [5,0], desired_speed = 8)
        area20 = desired_area(3, [32,agent2.position[0][1]])
        area21 = desired_area(4, [31.5, 0], last =  True)
        agent2.desired_area = area20
        agent2.desired_areas = [area20, area21]
        
        agent3 = agent(3, [12,34.5], [7,0], desired_speed = 10)
        area30 = desired_area(5, [31,agent2.position[0][1]])
        area31 = desired_area(6, [34.5,0], last =True)
        agent3.desired_area = area30
        agent3.desired_areas = [area30,area31]
        
        agent4 = agent(4, [37.5, 0], [0,10])
        area40 = desired_area(7, [agent4.position[0][0], 72], last = True)
        agent4.desired_area = area40
        agent4.desired_areas = [area40]
        
        agent5 = agent(5, [40.5,10],[0,5], desired_speed = 15)
        area50 = desired_area(8,[agent5.position[0][0], 31])
        area51 = desired_area(9, [0,40.5], last = True)
        agent5.desired_area = area50
        agent5.desired_areas = [area50, area51]
        
        agent6 = agent(6, [34.5,72], [0,-5])
        area60 = desired_area( 10, [agent6.position[0][0],36])
        area61 = desired_area(11, [72,34.5], last = True)
        agent6.desired_area = area60
        agent6.desired_areas = [area60, area61]
        
        agent7 = agent(7, [31.5,72], [0,-5], desired_speed = 10)
        area70 = desired_area(12, [34.5,0], last = True)
        agent7.desired_area = area70
        agent7.desired_areas = [area70]
        
        agent8 = agent(8, [72,40.5], [-5,0])
        area80 = desired_area(13, [0, agent8.position[0][1]], last = True)
        agent8.desired_area = area80
        agent8.desired_areas = [area80]
        
        agent9 = agent(9, [50,40.5], [-5,0], desired_speed = 13)
        area90 = desired_area(14, [0, agent9.position[0][1]], last = True)
        agent9.desired_area = area90
        agent9.desired_areas = [area90]
        
        agent10 = agent(10, [60,37.5], [-5,0])
        area100 = desired_area(15, [0, agent10.position[0][1]], last = True)
        agent10.desired_area = area100
        agent10.desired_areas = [area100]
        
        agents = [agent0, agent1, agent2, agent3, agent4, agent5, agent6, agent7,
                  agent8, agent9, agent10]
        t_end = 10
    return borders, agents, desired_areas, t_end, xlim, ylim
        

borders, agents, desired_areas, t_end, xlim, ylim = Scenarios(Scenario) 
 
t = np.arange(0, t_end, Delta_t)

##################### FUNCTIONS ##############################################

### AUXILIAR FUNCTIONS -------------------------------------------------------

def vector_norm(w):
    return np.sqrt(w[0]**2+w[1]**2)

def get_direction(w):
    return np.array(w)/vector_norm(w)

### SOCIAL FORCES ------------------------------------------------------------

def resultant_social_force(agent, agents, borders):
    
    ### Social Force of obstacle repulsion ---!
    def obstacle_repulsion(agent, borders):
        ra = agent.position_float
        Fab = np.array([0,0])
        
        for border in borders:
            rb, alpha = border.dist_border(ra)
             
            if 0 <= alpha <= 1:
                rab = ra - rb
                Fab = Fab + (beta_1 * np.exp(-(vector_norm(rab)/beta_2))
                       ) * get_direction(rab)
        return Fab
                    
    ### Social Force of agent repulsion ---!
    def agent_repulsion(agent, agents):
        fab = np.array([0,0])
        ra = agent.position_float
        va = agent.velocity_float
        for other in agents:
            if other.uid != agent.uid:
                rb = other.position_float
                vb = other.velocity_float
                rp = (rb-ra) + (vb - va) * Delta_t_prevision
                R = 2
                if vector_norm(rp)<R:
                    R = vector_norm(rp)-0.0001
                    print('Colidiu ', round(vector_norm(rp),2), 'm de distância')
                fab = fab + (-beta_1 * np.exp((-vector_norm(rp)+R)/beta_2) * 
                             get_direction(rp))
        return fab
    
    
    ### Social Force to atract the vehicle to the destiny ---!
    def target_force(agent):    
        if agent.target == True:
                SFt = (agent.desired_speed * agent.desired_direction() - agent.velocity_float)/tau
        else:
                SFt = np.array([0,0])
        return SFt
    
    ### Sum of all the forces mentioned ---!
    R = (obstacle_repulsion(agent, borders) + 
         agent_repulsion(agent, agents ) + 
         target_force(agent))
    
    return R



### 4 ORDER RUNGE KUTTA IMPLEMENTATION----------------------------------------

def motion():
    
    'Function to insert each k constants for Runge Kutta integration'
    def append_kx_and_kv(agent):
        R = resultant_social_force(agent, agents, borders)
        kx = agent.velocity_float
        kv = R
        agent.kx.append(kx)
        agent.kv.append(kv)
    
    'Loop to run all instants defined in the time vector'
    for i in t:

        'for loop to get k1'
        for agent in agents: 
            append_kx_and_kv(agent)
            
        'turn all parameters to get k2, in f(r,v) for t_i+1= t_i + Delta_t/2'
        for agent in agents:
            k1x = agent.kx[0]
            k1v = agent.kv[0]
            agent.velocity_float = agent.velocity[-1] + (Delta_t/2) *  k1v
            agent.position_float = agent.position[-1] + (Delta_t/2) * k1x
            
        'for loop to get k2'
        for agent in agents: 
            append_kx_and_kv(agent)
            
        'turn all parameters to get k3, in f(r,v) for t_i+1= t_i + Delta_t/2, \
        but now with the k2 constant'
        for agent in agents:
            k2x = agent.kx[1]
            k2v = agent.kv[1]
            agent.velocity_float = agent.velocity[-1] + (Delta_t/2) *  k2v
            agent.position_float = agent.position[-1] + (Delta_t/2) * k2x
            
            
        'for loop to get k3'
        for agent in agents: 
            append_kx_and_kv(agent)
            
        'Turn all parameters to get k4, in f(r,v) for t_i+1= t_i + Delta_t \
        but now with the k3 constant'
        for agent in agents:
            k3x = agent.kx[2]
            k3v = agent.kv[2]
            agent.velocity_float = agent.velocity[-1] + (Delta_t/2) *  k3v
            agent.position_float = agent.position[-1] + (Delta_t/2) * k3x
            
        'for loop to get k4'
        for agent in agents: 
            append_kx_and_kv(agent)
            
        'for loop to calculate the k factor for Runge Kutta integration, and \
        update the position and velocity of these agents'
        for agent in agents:
            k1x = agent.kx[0]
            k2x = agent.kx[1]
            k3x = agent.kx[2]
            k4x = agent.kx[3]

            k1v = agent.kv[0]
            k2v = agent.kv[1]
            k3v = agent.kv[2]
            k4v = agent.kv[3]
            
            kx = ( k1x + 2*k2x + 2*k3x + k4x ) / 6
            kv = ( k1v + 2*k2v + 2*k3v + k4v ) / 6
            agent.kx = []
            agent.kv = []
            
            'check if the agent already achived its target'
            if agent.active == True:
                v_new = agent.velocity[-1] + Delta_t * kv
                r_new = agent.position[-1] + Delta_t * kx
            elif agent.active == False:
                v_new = np.array([0,0])
                r_new = agent.position[-1]
                
            'update the values of position, velocity, and resultant'
            agent.velocity.append(v_new)
            agent.position.append(r_new)
            agent.resultant.append(kv)
            agent.position_float = r_new
            agent.velocity_float = v_new
        
        
### VISUALIZATION ------------------------------------------------------------

def get_values_to_plot(List):
    X = []
    Y = []
    norm = []
    for i in range(len(List)):
        X.append(List[i][0])
        Y.append(List[i][1])
        norm.append(vector_norm(List[i]))
    return X, Y

class visualization:
    def __init__(self, show_desired_areas = False):
        self.show_desired_areas = show_desired_areas
        
    'one view of the simulation'
    def basic_view(self) :
    
        # Plotting borders
        for border in borders:
            plt.plot(border.Qx, border.Qy, color = 'r')
        
        # Plotting the agents trajectories
        for agent in agents:
            X, Y = get_values_to_plot(agent.position)
            plt.plot(X, Y, ":")
    
            # Plotting the Resultant Force
            Rx, Ry = get_values_to_plot(agent.resultant)
            for i in range(len(t)):
                if t[i]%5 == 0: 
                    plt.quiver(X[i], Y[i], Rx[i], Ry[i], color = 'b', minshaft = 4, scale = 7)
        plt.savefig("visualizacao_completa_cenario"+str(Scenario)+".png")
    
    def iteration_view(self):
        fig, axs = plt.subplots(a, b, sharex=True, sharey=True, squeeze=False)
        plt.figure(dpi = 300)
        
        for v in range(a*b): 
            u = int(((v)/(a*b))*len(t))
            
            for agent in agents:
                ax = axs[v//b, v-b*(v//b)]
                x, y = get_values_to_plot(agent.position)
                
                'plotting the trajectorie in each step'
                ax.plot(x[:u], y[:u], ":")
                
                'ploting the speed of the agent'
                area = vector_norm(agent.velocity[u])
                ax.scatter(x[u], y[u], s=area,  alpha=0.5)
                
                ax.set_title("t = "+ str(round(t[u],1))+ " s",
                              fontsize=7, loc='right')
        
                
        for ax in fig.get_axes():
            # Plotting borders
            for border in borders:
                ax.plot(border.Qx, border.Qy, color = 'r')
            
            # Plotting desired areas
            for desired_area in desired_areas:
                r = desired_area.radius
                x, y = desired_area.position
                draw_circle = plt.Circle((x, y), r, alpha = 0.3, color = 'r')
                ax.set_aspect(1)
                ax.add_artist(draw_circle)
            if ylim != (0,0):
                plt.ylim(ylim)
            if xlim != (0,0):
                plt.xlim(xlim)
            ax.grid()
        fig.savefig("visualizacao_iterada_cenario"+str(Scenario)+".pdf")
    
    def speed_show(self):
            
        fig, ax = plt.subplots()
        plt.figure(dpi = 300)
        for agent in agents:
            vx, vy = get_values_to_plot(agent.velocity)
            speed = np.zeros(len(vx))
            for i in range(len(vx)):
                speed[i] = np.sqrt(vx[i]**2 + vy[i]**2)
                
            ax.plot(t, speed[:len(t)])
            ax.set_xlabel("time (s)")
            ax.set_ylabel("speed (m/s)")
        fig.savefig("visualizacao_velocidades_cenario"+str(Scenario)+".pdf")
        
    def gif(self):
        filenames = []
        for i in range(len(t)):
            if t[i] % 0.5 == 0:
                if ylim != (0,0):
                    plt.ylim(ylim)
                if xlim != (0,0):
                    plt.xlim(xlim)
                
                
                for agent in agents:
                    # plot the line chart
                    x, y = get_values_to_plot(agent.position)
                    
                    'plotting the trajectorie in each step'
                    plt.plot(x[:i], y[:i], ":")
                    
                    'ploting the speed of the agent'
                    area = vector_norm(agent.velocity[i])
                    plt.scatter(x[i], y[i], s=area,  alpha=0.5)
                
                for border in borders:
                    plt.plot(border.Qx, border.Qy, color = 'r')
                    
                # for desired_area in desired_areas:
                #     r = desired_area.radius
                #     x, y = desired_area.position
                #     draw_circle = plt.Circle((x, y), r, alpha = 0.3, color = 'r')
                #     plt.set_aspect(1)
                #     plt.add_artist(draw_circle)
                
                plt.title("t = "+ str(round(t[i],1))+ " s",
                             fontsize=8, loc='right')
                plt.grid()
                
                
                # create file name and append it to a list
                filename = f'{i}.png'
                filenames.append(filename)
                
                # save frame
                plt.savefig(filename)
                plt.close()
        # build gif
        with imageio.get_writer('motionScenario'+str(Scenario)+'ComTrajetoria.gif', mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
                
        # Remove files
        for filename in set(filenames):
            os.remove(filename)

        
  
start = time.time()              
motion()
end = time.time()
print('Processing time = ', round(end-start,3), 's')

visualization = visualization()

# visualization.basic_view()
visualization.iteration_view()
visualization.speed_show()
# visualization.gif()

############### DATA #########################################################
average_speed = []
for agent in agents:
    
    vx, vy = get_values_to_plot(agent.velocity)
    speed = np.zeros(len(vx))
    for i in range(len(vx)):
        speed[i] = np.sqrt(vx[i]**2 + vy[i]**2)
    
    average = sum(speed)/len(speed)
    average_speed.append(average)
    














