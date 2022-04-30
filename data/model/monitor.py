import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style 

style.use('fivethirtyeight')

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    try:
        graph_data = open("log/log.txt",'r').read()
    except:
        graph_data = open("model/log/log.txt",'r').read()
    
    lines = graph_data.split("\n")
    xs = []
    ys = []
    
    for line in lines:
        if len(line)>1:
            # graph_data = open("log/log.txt",'r').read()
            # lines = graph_data.split("\n")
            
            x, y = line.split(',')
            # x, y = float(x), float(y)
            # xs.append((float(x)/50))
            # ys.append(float(y)/float(x))
            # ys.append((float(y)/(float(x)/+1.0(float(x)/2.0))+1.0))
            # if float(x) < 1:
            #     # print(x)
            #     x *= 100
            #     # print(x)
            # if float(x) < 0.001:
            #     x /= 100
                
            xs.append(float(x))
            ys.append(float(y))
            # print(f'xy {ys} xs {xs}')
            
    ax1.clear()
    ax1.plot(xs,ys)
    
ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()