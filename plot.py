import torch

import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.manifold import TSNE



#print(input)

def log_average_weighted(input, ranking,  base = 1.8):
    weighted_average = np.zeros(input[0].shape)
    sum = 0
    for a,i in enumerate(ranking):
        element = input[i]
        element2 = np.asarray([math.log(a) for a in element])
        weighted_average += element2 * math.pow(base, -1*a) /2
        sum += math.pow(base, -1*a) /2
    weighted_average = weighted_average / sum
    weighted_average = np.asarray([math.exp(a) for a in weighted_average])
    
    return weighted_average

def log_average(input):
    weighted_average = np.zeros(input[0].shape)
    sum = 0
    for element in input:
        element2 = np.asarray([math.log(a) for a in element])
        weighted_average += element2 
        sum += 1
    weighted_average = weighted_average / sum
    weighted_average = np.asarray([math.exp(a) for a in weighted_average])
    
    return weighted_average

def process_log(input):
    accuracies1 = []
    accuracies2 = []
    accuracies3 = []
    lrs =[]
    lr = []
    max_lrs = 10
    for i,element in enumerate(input):
        #print(i, element)
        if (i +13) % 13 == 0 :
             accuracies1.append(element)
        elif (i +12) % 13 == 0 :
             accuracies2.append(element)
        elif (i +11) % 13 == 0 :
             accuracies3.append(element)
        else:
            lr.append(element)
            if len(lr) >= 10:
                lrs.append(lr)
                lr = []
                
    lrs = np.asarray(lrs)
    lrs = lrs[:,:max_lrs]
    
    rating = np.asarray(accuracies2) + np.asarray(accuracies3)
#     print(rating)
    ranking = np.argsort(rating)
    ranking = np.flip(ranking)
    print(ranking)
    
    
    return ranking, accuracies1,  accuracies2, accuracies3, lrs

def plot_lr_rate(input):
    base = 1.8
    ranking, accuracies1,  accuracies2, accuracies3, lrs = process_log(input)
    weighted_average = log_average_weighted(lrs, ranking, base = base)
    weighted_var = np.zeros(lrs[0].shape)
   # for i,element in enumerate(lrs):#
    
  #  for i,element in enumerate(lrs):
    sum = 0
    for a,i in enumerate(ranking):
        element = lrs[i]
      #  element = np.asarray([math.log(a) for a in element])
        weighted_var += np.asarray([math.log(a) for a in abs(weighted_average-element)]) * math.pow(base, -1* a) /2
      #  weighted_var += weighted_average-element * math.pow(2, -1* ranking[i]) /2
        sum += math.pow(base, -1*a) /2
    weighted_var = weighted_var / sum
    weighted_var = np.asarray([math.exp(a) for a in weighted_var])
    print(weighted_average)
    print(weighted_var)
    print(lrs[ranking[0]])
    
    x = [i for i in range(lrs[0].shape[0])]
    plt.plot(x,weighted_average, color = (0,0.1,1), label = "weighted average")
    plt.fill_between(x, [max(a,1e-7) for a in weighted_average-weighted_var], weighted_average+weighted_var, color = (0,0.1,1, 0.3), label = "deviation")
    plt.plot(x,lrs[ranking[0]], label = "best")
    plt.legend()
    plt.yscale('log')
    # for i,lr in enumerate(lrs):
    #     plt.plot(lr,color = (1,0,0, 1-ranking[i]/ranking.shape[0]))
    
    plt.show()

def combine_multiple(inputs):
    averages =[]
    for input in inputs:
        ranking, accuracies1,  accuracies2, accuracies3, lrs = process_log(input)
        weighted_average = log_average_weighted(lrs, ranking, base = 1.8)
        x = [i for i in range(weighted_average.shape[0])]
  #      plt.plot(x,weighted_average)
        averages.append(weighted_average)
    average = log_average(averages)
    
    print("average over all results",np.asarray(average))
    
    plt.plot(x,[2e-5 for a in x], label = "baseline")
    plt.plot(x,average, color = (0,0.1,1), label = "combined learning rate")
    plt.legend()
    plt.xlabel("choice")
    plt.ylabel("learning rate")
    plt.yscale('log')
    plt.show()

def plot_TSNE_clustering(X,y = None):
   # X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    mean_vec = np.mean(vectors, axis = 0)
    X.append(mean_vec)
    X_embedded = TSNE(n_components=2,init='random').fit_transform(X)
    color = np.zeros(len(X))
    color[-1] = 1
    plt.scatter(X_embedded[:,0], X_embedded[:,1], c = color)
    plt.show()
    
if __name__ == '__main__': 
    from ete3 import Tree
    t = Tree( "((a,b),c);" )
    t.show()



    pretrained = torch.load("results/sst2/pretrained.pt")
    afterfine = torch.load("results/sst2/afterfine.pt")
    word_freq = torch.load("results/sst2/baseword_freq.pt")
    pretrained = pretrained[:,:,word_freq > 10]
    afterfine = afterfine[:,:,word_freq > 10]
    word_freq = word_freq[word_freq > 10]
    pretrained = pretrained / word_freq
    afterfine = afterfine / word_freq
    changes = torch.abs(pretrained -afterfine)
    print(changes.shape)

    changessorted = np.argsort(np.asarray(changes.cpu())[0,0,:], axis = 0)
    #pretrained = np.argsort(np.asarray(pretrained)[:,1:], axis = 1)
   # afterfine = np.argsort(np.asarray(afterfine)[:,1:], axis = 1)
   # print(changessorted.shape)
  #  print(changessorted)
    

  #  print(pretrained)
   # print(afterfine)

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    words = tokenizer.decode(changessorted[-200:], clean_up_tokenization_spaces = False,skip_special_tokens = False, spaces_between_special_tokens=True ).split(" ")
  #  print(len(words))
    words = [w for w in words if w.isalpha() and len(w)> 3]
    joinedwords = ", ".join(words)
    print(joinedwords)
    #print(tokenizer.batch_decode(afterfine[:1,-200:], clean_up_tokenization_spaces = False))

    from gensim.models import KeyedVectors
    import gensim.downloader

    # Load vectors directly from the file
    model = gensim.downloader.load('word2vec-google-news-300')

    vectors = [model[w] for w in words if w in model]

  #  print(vectors)
    
    mean_vec = np.mean(vectors, axis = 0)
    plot_TSNE_clustering(vectors)
   # print(mean_vec)

    print(model.most_similar(positive = [mean_vec], topn = 5))
































