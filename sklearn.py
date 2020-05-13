import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


def reg():
    boston=load_boston()
    bos=pd.DataFrame(boston.data)
    bos.col=boston.feature_names
    bos['PRICE'] = boston.target
    X=bos.drop('PRICE',axis=1)
    linereg=LinearRegression()
    linereg.fit(X,bos['PRICE'])
    #calculate coefficient of each factor
    coef=(np.fabs(linereg.coef_)).tolist()

    max_index=coef.index(max(coef))
    return boston.feature_names[max_index]
        
def kmean(clusters):
    # create Kmean model for iris data
    iris=load_iris()
    iris_data=iris['data'] 
    scale = MinMaxScaler().fit(iris_data)
    iris_dataScale = scale.transform(iris_data)
    kmeans = KMeans(n_clusters=clusters,random_state=123).fit(iris_dataScale)
    print(kmeans)

    #plot the clusters
    tsne = TSNE(n_components=2,init='random',random_state=177).fit(iris_data) 
    df=pd.DataFrame(tsne.embedding_)
    df['labels']=kmeans.labels_ 
    if clusters==3:
        df1 = df[df['labels']==0]
        df2 = df[df['labels']==1]
        df3 = df[df['labels']==2]
        plt.plot(df1[0],df1[1],'bo',df2[0],df2[1],'r*',df3[0],df3[1],'gD')
        plt.show()
        
    if clusters==4:
        df1 = df[df['labels']==0]
        df2 = df[df['labels']==1]
        df3 = df[df['labels']==2]
        df4 = df[df['labels']==3]
        plt.plot(df1[0],df1[1],'bo',df2[0],df2[1],'r*',df3[0],df3[1],'gD',df4[0],df4[1],'m<')
        plt.show()
        
    if clusters==5:
        df1 = df[df['labels']==0]
        df2 = df[df['labels']==1]
        df3 = df[df['labels']==2]
        df4 = df[df['labels']==3]
        df5 = df[df['labels']==4]
        plt.plot(df1[0],df1[1],'bo',df2[0],df2[1],'r*',df3[0],df3[1],'gD',df4[0],df4[1],'m<',df5[0],df5[1],'yo',)
        plt.show()

if __name__ == '__main__':
    print(reg())
    kmean(5)


