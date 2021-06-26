import numpy as np
import numpy.linalg as LA

def topk(ma1,gip,nei):
    for i in range(ma1.shape[0]):
        ma1[i,i]=0
        gip[i,i]=0
    ma=np.zeros((ma1.shape[0],ma1.shape[1]))
    for i in range(ma1.shape[0]):
        if sum(ma1[i]>0)>nei:
            yd=np.argsort(ma1[i])
            ma[i,yd[-nei:]]=1
            ma[yd[-nei:],i]=1
        else:
            yd=np.argsort(gip[i])
            ma[i,yd[-nei:]]=1
            ma[yd[-nei:],i]=1
    return ma

def GIP(A):#A is numpy 2D array
    gamad1=1
    sumk1=0
    ss=A.shape[1]
    for nm in range(ss):
        sumk1=sumk1+LA.norm(A[:,nm],ord=2)**2
    gamaD1=gamad1*ss/sumk1
    KD=np.mat(np.zeros((ss,ss)))
    for ab in range(ss):
        for ba in range(ss):
            KD[ab,ba]=np.exp(-gamaD1*LA.norm(A[:,ab]-A[:,ba])**2)
    gamad2=1
    sumk2=0
    mm=A.shape[0]
    for mn in range(mm):
        sumk2=sumk2+LA.norm(A[mn,:],ord=2)**2
    gamaD2=gamad2*mm/sumk2
    KM=np.zeros((mm,mm))
    for cd in range(mm):
        for dc in range(mm):
            KM[cd,dc]=np.exp(-gamaD2*LA.norm(A[cd,:]-A[dc,:])**2)
    return np.array(KM),np.array(KD)