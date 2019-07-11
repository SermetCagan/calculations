import numpy as np
from math import  log

x16=np.zeros(16,float)
w16=np.zeros(16,float)

x16[8]=9.50125098376374402e-02;      w16[8]=1.89450610455068496E-0001
x16[9]=2.81603550779258913E-0001;    w16[9]=  1.82603415044923589E-0001
x16[10]=4.58016777657227386E-0001;   w16[10]= 1.69156519395002538E-0001
x16[11]=6.17876244402643748E-0001;   w16[11]=1.49595988816576732E-0001
x16[12]=7.55404408355003034E-0001;   w16[12]=1.24628971255533872E-0001
x16[13]=8.65631202387831744E-0001;   w16[13]=   9.51585116824927848E-0002
x16[14]=9.44575023073232576E-0001;   w16[14]=  6.22535239386478929E-0002
x16[15]=9.89400934991649933E-0001;   w16[15]=  2.71524594117540949E-0002;

#r_log=np.array([0.041448, 0.245275, 0.556165, 0.848982])
#w_log=-np.array([0.383464, 0.386875, 0.190435, 0.039225])

r_log=np.array([0.041448480199322, 0.245274914320325, 
                0.556165453559947, 0.848982394532840])
w_log=np.array([-0.383464068144763, -0.386875317774810, 
                -0.190435126950392, -0.039225487130036]);

for k in range(8):
    x16[7-k]=-x16[8+k]
    w16[7-k]=w16[8+k]
    
def calculat_all(n,a,b):
    w=np.zeros(n,float)
    r=np.zeros(n,float)
    ratio=16.0/n
    #x=np.linspace(a,b,int(n/16))
    jacobian=(b-a)/2.0
    for k in range(int(n/16)):
        r[k*16:(k+1)*16]=a+(b-a)*k*ratio+jacobian*(x16[:]+1)*ratio
        w[k*16:(k+1)*16]=jacobian*w16[:]*ratio
    return (r,w)

def calculat_all2(n,a,b,c):
    (r1,w1)=calculat_all(n,a,b)
    (r2,w2)=calculat_all(n,b,c)
    #r=np.array(list(r1)+list(r2))
    r=np.hstack((r1, r2))
    #w=np.array(list(w1)+list(w2))
    w=np.hstack((w1, w2))
    return (r, w)

def int_log(): 
    return(r_log, w_log)

def int_log2(k, a, b, c, n):
    (r1, w1)=int_log()
    r1=(b-a)*r1+a
    w1=abs(b-a)*w1
    w1=w1/np.log(k*abs(r1-a))
   
    (r2, w2)=calculat_all(16, 0.0, 1.0)
    #r2_in=np.array(r2)
    r2=(b-a)*r2+a
    w2=abs(b-a)*log(k*abs(b-a))*w2/np.log(k*abs(r2-a))
    #w2=(b-a)*log(k*(b-a))*w2/np.log(k*r2)
    r=np.hstack((r1, r2))
    w=np.hstack((w1, w2))
    (r3, w3)= calculat_all(n, b, c)
    r=np.hstack((r, r3))
    w=np.hstack((w, w3))
    return (r, w)

def integrat_singularity_right(a,b):
    (root_long,weight_long)=calculat_all(16,-1.0,1.0)
    xi=np.zeros(8,'float')
    #weight_final=np.zeros(8,'float')
    weight=np.zeros(8,'float')
    xi[:]=root_long[8:]
    weight[:]=weight_long[8:]
    root=a+(b-a)*(1.0-xi*xi)
    #weight_final=(b-a)*xi*xi*weight
    return(root,weight)

def calculat_all3(n,a,b,c):
    (r1,w1)=calculat_all(n,a,b)
    (r2,w2)=integrat_singularity_right(b,c)
    r=np.hstack((r1,r2))
    w=np.hstack((w1,w2))
    return (r,w)
    


        
    
  
