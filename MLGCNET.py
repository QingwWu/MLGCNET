import numpy as np
import random
from sklearn.ensemble import ExtraTreesClassifier
from process import topk,GIP
from emb import calemb
from argparse import  ArgumentParser


def parse_args():
    parser = ArgumentParser(description='MLGCN Example')
    parser.add_argument('--dimensions', default=128, type=int, help='the dimensions of embedding for each node.')
    parser.add_argument('--epochs', default=1000, type=int, help='The training epochs of multi_layer GCN') 
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')   
    parser.add_argument('--dropout', default=0.4, type=float, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--adj_dropout', default=0.6, type=float, help='Adjacency matrix dropout rate (1 - keep probability).')    
    parser.add_argument('--seed',default=1, type=int, help='random seed (default: 1)')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    
    ########## load data #########
    LD=np.loadtxt('lda.csv', delimiter=',')
    sl1=np.loadtxt('sl.csv', delimiter=',') 
    sd1=np.loadtxt('sd.csv', delimiter=',')
    
    ################## training data ###########   
    l=LD.shape[0]
    d=LD.shape[1]
    knownIndices=[]
    for ind in range(l*d):
        if (ind%d==0):
            j=0
        else:
            j=ind%d
        i=int(ind/d)
        if (LD[i,j]==1):
            knownIndices.append(ind)     
    allIndices=list(range(l*d))
    allIndices1=list(set(allIndices).difference(set(knownIndices)))
    negativeIndices=random.sample(allIndices1,len(knownIndices))
    positiveAndNegativeIndices=knownIndices+negativeIndices
    
    ################## embedding ###################    
    sl2,sd2=GIP(LD)
    sl=topk(sl1,sl2,5)
    sd=topk(sd1,sd2,5) 
    args = parse_args()
    epochs = args.epochs
    dim= args.dimensions
    lr = args.lr
    dp = args.dropout
    adjdp = args.adj_dropout
    seed = args.seed  
    embedall= calemb(LD, sl, sd, seed, epochs, dim, dp, lr, adjdp)
    slemb=embedall[0:l,:]
    sdemb=embedall[l:,:]          
            
    ######################## training feature ####################################
    positiveAndNegativelncIndices=[]
    positiveAndNegativedisIndices=[]             
    for inde in range(len(positiveAndNegativeIndices)):
        if (positiveAndNegativeIndices[inde]%d==0):
            l=0
        else:
            l=positiveAndNegativeIndices[inde]%d
        k=int(positiveAndNegativeIndices[inde]/d)
        positiveAndNegativelncIndices.append(k)
        positiveAndNegativedisIndices.append(l) 
    foldFeaturelnc = slemb[positiveAndNegativelncIndices]
    foldFeaturedis = sdemb[positiveAndNegativedisIndices]        
    foldFeature = np.hstack((foldFeaturelnc,foldFeaturedis))
    ####################### training lable ####################################
    knowlabel=np.ones((len(knownIndices),1))
    negativelabel=np.zeros((len(negativeIndices),1))
    y_train=np.concatenate((knowlabel,negativelabel),axis=0)
    
    ############ ET predict ##########################
    ET = ExtraTreesClassifier(n_estimators=500)
    ET.fit(foldFeature, y_train.ravel())
    gpositiveAndNegativelncIndices=[]
    gpositiveAndNegativedisIndices=[]
    for ginde in range(len(allIndices1)):
        if (allIndices1[ginde]%d==0):
            gl=0
        else:
            gl=allIndices1[ginde]%d
        gk=int(allIndices1[ginde]/d)
        gpositiveAndNegativelncIndices.append(gk)
        gpositiveAndNegativedisIndices.append(gl)
    globalFeaturelnc = slemb[gpositiveAndNegativelncIndices]
    globalFeaturedis = sdemb[gpositiveAndNegativedisIndices]
    globalFeatureVectors=np.hstack((globalFeaturelnc,globalFeaturedis))
    y_pred_proba = ET.predict_proba(globalFeatureVectors)[:, 1]
    
    ############## Probability of all unknown lncRNA-disease associations#########
    Allscore=np.zeros((l,d))
    for i in range (len(gpositiveAndNegativelncIndices)):
        Allscore[gpositiveAndNegativelncIndices[i],gpositiveAndNegativedisIndices[i]] = y_pred_proba[i]
        


        
        
