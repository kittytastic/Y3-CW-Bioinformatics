import matplotlib.pyplot as plt
import EM

if __name__ =="__main__":
    observedSeq = [0,0,0,1,1,2,3,1,1,1,1,0,3,3,3,3,3,3,3,1,1,1,1,1,1,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,0,1,1,1,1,1,2,1,2,1,2,1,2,1,2,1,2,0,0,0,1,1,2,1,1,1,1,1,1,1,1,1,4,4,4,4]
    #observedSeq = [0,0,0,1,1,2,3,1,1,1,1,0,3,3,3,3,3,3,3,1,1,1,1,1,1,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,0,1,1,1,1,1,2,1,2,1,2,1,2,1,2,1,2,0,0,0,1,1,2,1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,2,3,1,1,1,1,0,3,3,3,3,3,3,3,1,1,1,1,1,1,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,0,1,1,1,1,1,2,1,2,1,2,1,2,1,2,1,2,0,0,0,1,1,2,1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,2,3,1,1,1,1,0,3,3,3,3,3,3,3,1,1,1,1,1,1,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,0,1,1,1,1,1,2,1,2,1,2,1,2,1,2,1,2,0,0,0,1,1,2,1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,2,3,1,1,1,1,0,3,3,3,3,3,3,3,1,1,1,1,1,1,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,0,1,1,1,1,1,2,1,2,1,2,1,2,1,2,1,2,0,0,0,1,1,2,1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,2,3,1,1,1,1,0,3,3,3,3,3,3,3,1,1,1,1,1,1,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,0,1,1,1,1,1,2,1,2,1,2,1,2,1,2,1,2,0,0,0,1,1,2,1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,2,3,1,1,1,1,0,3,3,3,3,3,3,3,1,1,1,1,1,1,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,0,1,1,1,1,1,2,1,2,1,2,1,2,1,2,1,2,0,0,0,1,1,2,1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,2,3,1,1,1,1,0,3,3,3,3,3,3,3,1,1,1,1,1,1,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,0,1,1,1,1,1,2,1,2,1,2,1,2,1,2,1,2,0,0,0,1,1,2,1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,2,3,1,1,1,1,0,3,3,3,3,3,3,3,1,1,1,1,1,1,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,0,1,1,1,1,1,2,1,2,1,2,1,2,1,2,1,2,0,0,0,1,1,2,1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,2,3,1,1,1,1,0,3,3,3,3,3,3,3,1,1,1,1,1,1,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,0,1,1,1,1,1,2,1,2,1,2,1,2,1,2,1,2,0,0,0,1,1,2,1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,2,3,1,1,1,1,0,3,3,3,3,3,3,3,1,1,1,1,1,1,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,0,1,1,1,1,1,2,1,2,1,2,1,2,1,2,1,2,0,0,0,1,1,2,1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,2,3,1,1,1,1,0,3,3,3,3,3,3,3,1,1,1,1,1,1,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,0,1,1,1,1,1,2,1,2,1,2,1,2,1,2,1,2,0,0,0,1,1,2,1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,2,3,1,1,1,1,0,3,3,3,3,3,3,3,1,1,1,1,1,1,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,0,1,1,1,1,1,2,1,2,1,2,1,2,1,2,1,2,0,0,0,1,1,2,1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,2,3,1,1,1,1,0,3,3,3,3,3,3,3,1,1,1,1,1,1,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,0,1,1,1,1,1,2,1,2,1,2,1,2,1,2,1,2,0,0,0,1,1,2,1,1,1,1,1,1,1,1,1,0,0,0,1,1]
   
    #observedSeq = [0,0,0,1,1]
    #observedSeq = [0,1,2,3,4]*40
    
    print(len(observedSeq))
    hidden = 10
    observeable = 5

    for i in range(5):
        _, _, _, probs = EM.estimateModel(observedSeq, hidden*(i+1), observeable, verbose=True) 
        plt.plot(probs, color='C%d'%i)
    plt.ylabel('$p(x|\lambda)$')
    plt.yscale('log')

    plt.xlabel('Model iteration')
    plt.savefig("temp.png")