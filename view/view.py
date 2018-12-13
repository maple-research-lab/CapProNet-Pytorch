import numpy as np
from scipy.linalg import fractional_matrix_power
import matplotlib.pyplot as plt

features = np.load('features.npy')
labels = np.load('labels.npy')
weights = np.load('weights.npy')

for cls in range(10):
    fig = plt.figure(cls+1)
    
    w = np.transpose(weights[cls*2:(cls+1)*2])
    proj = fractional_matrix_power(np.matmul(np.transpose(w), w), -0.5)
    proj = np.matmul(proj, np.transpose(w))
    
    for ii in range(500):
        x = features[ii][:, None]
        u = np.matmul(proj, x)
        if labels[ii]==cls:
            plt.plot(u[0], u[1], 'ro')
        else:
            plt.plot(u[0], u[1], 'g^')
    plt.title('Capsule subspace '+str(cls))
    ax = fig.add_subplot(1, 1, 1)

    ax.spines['left'].set_position(('data', 0.0))
    ax.spines['bottom'].set_position(('data', 0.0))
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_axisbelow(False)
    
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()
    plt.show()

#w = np.transpose(weights[:2])
#proj = fractional_matrix_power(np.matmul(np.transpose(w), w), -0.5)
#proj = np.matmul(proj, np.transpose(w))
#for ii in range(300):
#    x = features[ii][:, None]
#    u = np.matmul(proj, x)
#    if labels[ii]==0:
#        plt.plot(np.abs(u[0]), np.abs(u[1]), 'ro')
#    else:
#        plt.plot(np.abs(u[0]), np.abs(u[1]), 'g^')
#    print(u)
#    print(labels[ii])
#plt.show()