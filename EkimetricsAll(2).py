import csv
import pandas
import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

'''
\*ne marche pas
column1={}
for h in headers1:
    column1[h]=[]
column2={}
for h in headers2:
    column2[h]=[]
print column1
print column2

iter1=reader1
iter2=reader2
for row in iter1:
    for(h,r) in zip(headers1,row):
        column1[h].append(r)
for row in iter2:
    for(h,r) in zip(headers2,row):
        column2[h].append(r)
print column1['Quantity']  
*\  
'''

def test_generation(transactions,refTrans, client_index, clients_ind):
    print("Generation du test...")
    test=list()
    i=0
    while (i<len(transactions)):
        l=[]
        client=transactions[i][0]
        if(clients_ind[client_index[i]]==0):
            code=transactions[i][13]
            j=i
            while(j<len(transactions) and transactions[j][13]==code):
                j+=1
            if(j<len(transactions) and transactions[j][0]==client):
                while(j<len(transactions) and transactions[j][0]==client):
                    l.append(refTrans[j])
                    j+=1
                test.append(l)
            else:
                test.append([])
            i+=1
        else:
            j=i
            while(j<len(transactions) and transactions[j][0]==client):
                test.append([])
                j+=1
            i=j
    return test
    
def sales(transactions,refAll):
    sales=[0]*len(refAll)
    for i in range(len(transactions)):
        if(clients_ind[client_index[i]]==1):
            x=refTrans[i]
            sales[refAll.index(x)]+=1
    return sales
    
def matrix_generation_dblist(transactions, refTrans, refAll):
    print("Generation de la matrice...")
    graph=list(list())
    for i in range(len(refAll)):
        graph.append([0]*len(refAll))
    i=0
    while (i<len(transactions)):
        l=[]
        client=transactions[i][0]
        while(i<len(transactions) and transactions[i][0]==client):
            l.append(refTrans[i])
            i+=1
        for ref in range(len(l)):
            for ref2 in range(ref):
                    x=refAll.index(l[ref])
                    y=refAll.index(l[ref2])
                    if x!=y:
                        graph[x][y]+=1
                        graph[y][x]+=1
    return graph
    
def matrix_generation_numpy(transactions, refTrans, refAll, client_index, clients_ind):
    print("Generation de la matrice...")
    graph=np.zeros((len(refAll),len(refAll)),dtype=int)
    i=0
    while (i<len(transactions)):
        l=[]
        client=transactions[i][0]
        if(clients_ind[client_index[i]]==1):
            while(i<len(transactions) and transactions[i][0]==client):
                l.append(refTrans[i])
                i+=1
            for ref in range(len(l)):
                for ref2 in range(ref):
                        x=refAll.index(l[ref])
                        y=refAll.index(l[ref2])
                        if x!=y:
                            graph[x][y]+=1
                            graph[y][x]+=1
        else:
            while(i<len(transactions) and transactions[i][0]==client):
                i+=1         
    return graph
    
   
def jaccard(graph,sales):
    jaccard=np.zeros(graph.shape,dtype=float)
    matA=np.repeat(np.array([sales]),len(sales),0)
    total=matA.T+matA
    jaccard=np.true_divide(graph,total)
    return jaccard
    

print("Ouverture des fichiers...")
prod=pandas.read_csv("Produits.csv",sep=";")
trans=pandas.read_csv("Short.csv",sep=";")
df=pandas.DataFrame(index=prod.Reference)

file1=open("Short.csv")
file2=open("Produits.csv")
reader1=csv.reader(file1,delimiter=";")
reader2=csv.reader(file2,delimiter=";")
headers1=next(reader1)
headers2=next(reader2)

transactions=list(reader1)
produits=list(reader2)
produits.pop()

'''
reference=list()
for row in produits:
    reference.append(row[11])
'''
    
'''
for row in transactions:
    n=len(row[3])
    while row[3][n-1]==' ':
        row[3]=row[3][:-1]
        n-=1
'''
        
# client_index : numero du client pour chaque transactions
# transclient : numero de la premiere transaction de chaque client
client_index=list()
transclient=list()
i=0
j=0
while(j<len(transactions)):
    transclient.append(j)
    k=j
    while (k<len(transactions) and transactions[j][0]==transactions[k][0]):
        client_index.append(i)
        k+=1
    j=k
    i+=1
    
refTrans=list()
for row in transactions:
    x=row[3]
    n=len(x)
    while x[n-1]==' ':
        x=x[:-1]
        n-=1
    refTrans.append(x)
    
refAll=list(set(refTrans))

print(len(refAll)) 
print(len(transclient))

clients_train, clients_test = train_test_split(range(len(transclient))) #splitter intelligemment

print(len(clients_train))

clients_ind=[0]*len(transclient)
for i in clients_train:
    clients_ind[i]=1
            
sales=sales(transactions,refAll)
test=test_generation(transactions,refTrans,client_index,clients_ind)
graph=matrix_generation_numpy(transactions,refTrans,refAll,client_index,clients_ind)
#test=test_generation(transactions,refTrans,client_index,[0 for x in transclient])#clients_ind)
#graph=matrix_generation_numpy(transactions,refTrans,refAll,client_index,[1 for x in transclient])#clients_ind)
jaccard=jaccard(graph,sales)

'''
file3=open("CommonSalesMatrix_refAll.csv")
graph=[list(map(int,rec)) for rec in csv.reader(file3,delimiter=",")]
print graph[0],graph[1]
'''

'''
d=csv.writer(open("refAll.csv","wb"))
d.writerow(refAll)
'''

print("Test...")
count=0
similarity=0
success=0
successMul=0
successBest=0
number=10
similarityJ=0
successJ=0
successMulJ=0
successBestJ=0
similarityBest=0
salesPropositions=[]
salesPropositionsJ=[]
BestSeller=refAll[sales.index(max(sales))]
print(max(sales))
print(BestSeller)
for i in range(len(transactions)):
    if test[i]!=[]:
        ref=refTrans[i]
        n=len(ref)
        count+=1
        index=refAll.index(ref)
        #mostCommon=graph[index].index(max(graph[index]))#faster in numpy:argmin
        mostCommon=graph[index].argmax()
        mostCommonJ=jaccard[index].argmax()
        proposition=refAll[mostCommon]
        propositionJ=refAll[mostCommonJ]
        salesPropositions.append(sales[mostCommon])
        salesPropositionsJ.append(sales[mostCommonJ])
        if (proposition[0:3]==ref[0:3]):
            similarity+=1
        if (propositionJ[0:3]==ref[0:3]):
            similarityJ+=1
        if (BestSeller[0:3]==ref[0:3]):
            similarityBest+=1
        #prop.sort(key=lambda x:graph[index][x],reverse=True)
        prop=graph[index].argsort()[::-1][:number]
        propJ=jaccard[index].argsort()[::-1][:number]
        b=False
        for j in range(number):
            if refAll[prop[j]] in test[i]:
                b=True
        if b==True:
            successMul+=1
        if proposition in test[i]:
            success+=1
        bj=False
        for j in range(number):
            if refAll[propJ[j]] in test[i]:
                bj=True
        if bj==True:
            successMulJ+=1
        if propositionJ in test[i]:
            successJ+=1
        if BestSeller in test[i]:
            successBest+=1
print("Succes avec une proposition:")
print(success,"/",count)
print(int(100*success/count),"%")
print("Succes avec",number," propositions:")
print(successMul,"/",count)
print(int(100*successMul/count),"%")
print("Propositions de la meme famille:")
print(similarity,"/",count)
print(int(100*similarity/count),"%")
print("Succes Jaccard avec une proposition:")
print(successJ,"/",count)
print(int(100*successJ/count),"%")
print("Succes Jaccard avec",number," propositions:")
print(successMulJ,"/",count)
print(int(100*successMulJ/count),"%")
print("Propositions Jaccard de la meme famille:")
print(similarityJ,"/",count)
print(int(100*similarityJ/count),"%")
print("Succes du best seller:")
print(successBest,"/",count)
print(int(100*successBest/count),"%")
print("Propositions de la meme famille que le best seller:")
print(similarityBest,"/",count)
print(int(100*similarityBest/count),"%")
#plt.scatter(range(len(sales)),sorted(sales))
'''
sortedSales=sorted(sales)
median=sortedSales[len(sales)/2]
average=sum(sales)/float(len(sales))
ligne=[average]*len(salesPropositions)
ligne2=[median]*len(salesPropositions)
truc=[100*float(x)/len(salesPropositions) for x in range(len(salesPropositions))]
plt.plot(truc,sorted(salesPropositions,reverse=True))
plt.plot(truc,ligne)
plt.plot(truc,ligne2)
ticks=[x * 10 for x in range(0, 10)]
plt.xticks(ticks)
plt.xlabel("%")
plt.ylabel("Nombre de ventes totales de la proposition")
plt.title("Are the propositions best sellers? (compared to median and average)")
#plt.show()
#plt.savefig('PropositionsNaives.png')
plt.savefig('PropositionsNaives.pdf')
'''

file1.close()
file2.close()
