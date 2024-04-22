import scipy as sp
import pydicom as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from PIL import Image
import math
import copy


# These are the problem I think

def forDerivX(u):
    u1 = copy.deepcopy(u)
    u2 = copy.deepcopy(u)
    # print(type(u))
    for i in u1:
        tempi = i[0]
        for j in range(0,len(i) -1):
            i[j] = i[j+1]
        i[len(i)-1] = tempi
    u1 = np.array(u1)
    u2 = np.array(u2)
    # print(u2)
    # print(u)
    return u1 - u2




def forDerivY(u):
    u1 = copy.deepcopy(u)
    u2 = copy.deepcopy(u)
    # print(u1)
    tempi = copy.deepcopy(u1[0])
    for i in range(0, len(u1)-1):
        u1[i] = u1[i + 1]
    u1[len(u1)-1] = tempi
    u1 = np.array(u1)
    u2 = np.array(u2)
    return u1 - u2
    

def backDerivX(u):
    u1 = copy.deepcopy(u)
    u2 = copy.deepcopy(u)
    # print(type(u))
    for i in u1:
        tempi = i[len(i) - 1]
        for j in range(len(i)-1, 0, -1):
            # print(j)
            i[j] = i[j-1]
        i[0] = tempi
    u1 = np.array(u1)
    # u2 = np.array(u2)
    # print(u2)
    # print(u1)
    return u2 - u1

def backDerivY(u):
    u1 = copy.deepcopy(u)
    u2 = copy.deepcopy(u)
    # print("In BX")
    # print(u1)
    # print(u2)
    tempi = copy.deepcopy(u1[len(u1)-1])
    for i in range(len(u1)-1, 0 , -1):
        u1[i] = u1[i - 1]
    # print("BREAK")
    # print(u2[0])
    # print(tempi)
    # print(tempi)
    u1[0] = tempi
    u1 = np.array(u1)
    u2 = np.array(u2)
    # print(u2)
    # print(u1)
    return u2 - u1





def main():
    # np.set_printoptions(precision=8, suppress=True, floatmode="fixed")
    
    # testArray = [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20], [21,22,23,24,25]]
    # print(testArray)
    # print(backDerivY(forDerivY(testArray)))
    # exit()

    # Reads in filename from current directory
    filename = "./" + str(sys.argv[1])
    # Imports image and uses greyscale


    img = Image.open(filename).convert('L')
    imgArray = np.array(img)

    print(imgArray.shape)

    padNum = 10

    padImg = np.pad(imgArray, padNum, mode="symmetric")

    print(padImg.shape)
    plt.imshow(padImg, cmap="gray")
    # plt.show()


    # padImg = padImg.transpose()

    w1 = np.zeros(padImg.shape)
    w2 = np.zeros(padImg.shape)

    b11 = np.zeros(padImg.shape)
    b12 = np.zeros(padImg.shape)

    v = np.zeros(padImg.shape)
    b2 = np.zeros(padImg.shape)

    alfa = 20
    beta = 20
    theta1 = 5
    theta2 = 5

    meshgridSizeX = np.arange(0, padImg.shape[1])
    meshgridSizeY = np.arange(0, padImg.shape[0])

    Y,X = np.meshgrid(meshgridSizeX, meshgridSizeY)
    # print(Y)
    # print(X)
    print("Size Y is " + str(Y.shape))
    print("Size X is " + str(X.shape))
    print("Size padImg is " + str(padImg.shape))

    # We're confident up to here


    # Seems to be the same
    G = np.cos(2*np.pi*X/padImg.shape[0]) + np.cos(2*np.pi*Y/padImg.shape[1]) - 2
    # print(G)

    # np.savetxt("g2.txt", np.round(G, 8))

    
    div_w_b2=backDerivX(w1-b11)+backDerivY(w2-b12)
    # np.savetxt("dwb2.txt", div_w_b2)
    for i in range (1, 50):
        print(i)
        div_w_b=backDerivX(w1-b11)+backDerivY(w2-b12)
        lap_v_b=backDerivX(forDerivX(v-b2))+backDerivY(forDerivY(v-b2))
        g=padImg-theta1*div_w_b+theta2*lap_v_b
        u=(np.fft.ifftn(np.fft.fftn(g)/(1-2*theta1*G+4*theta2*np.power(G,2)))).real
        

        np.savetxt("u1p.txt" , u)
        
        c1=forDerivX(u)+b11
        # np.savetxt("c1p.txt" , c1)
        c12=forDerivX(u)
        # np.savetxt("c12p.txt" , c12)
        c21=forDerivY(u)
        # np.savetxt("c21p.txt", c21)
        c2=forDerivY(u)+b12
        # np.savetxt("c2p.txt" , c2)

    
        abs_c=np.sqrt(np.power(c1,2)+np.power(c2,2)+np.spacing(1))
        # abs_c=np.sqrt(np.power(c1,2)+np.power(c2,2))
        # np.savetxt("abscp.txt" , abs_c)
        w1=np.multiply(np.maximum(np.subtract(abs_c,(alfa/theta1)),0), np.divide(c1, abs_c))
        # np.savetxt("w1p.txt" , w1)
        w2=np.multiply(np.maximum(np.subtract(abs_c,(alfa/theta1)),0), np.divide(c2, abs_c))
        # np.savetxt("w2p.txt" , w2)

        s=backDerivX(forDerivX(u))+backDerivY(forDerivY(u))+b2
        bxfx = backDerivX(forDerivX(u))
        byfy = backDerivY(forDerivY(u))
        # np.savetxt("bxfxp.txt" , bxfx)
        # np.savetxt("byfyp.txt" , byfy)
        # np.savetxt("sp.txt" , s)
        # np.savetxt("b2p.txt" , b2)

        v=np.maximum(np.abs(s)-beta/theta2,0)*np.sign(s)
        # np.savetxt("vp.txt" , v)

        b11=c1-w1
        b12=c2-w2
        b2=s-v

        # print("This is iteration " + str(i) + " and here is c1 " + str(c1))


    finishedImage = u[padNum:padImg.shape[0]-padNum, padNum:padImg.shape[1]-padNum]

    plt.imshow(u, cmap="gray")
    plt.show()




    



    
    

    


if __name__ == "__main__":
    main()