import  sys
import  math
import  matplotlib.pyplot as plt
import  numpy as np
from    sklearn.datasets import fetch_california_housing as dados

e = math.exp(1)

def log(x): return np.log(x)

def theta(z):
    '''
    ENTRADA:
            int z - um inteiro resultante do produto escalar w^{T} * \tilde{x}

    FUNCIONAMENTO:
            o método retornará o valor da função sigmoide \theta
            calculada num valor z

    SAÍDA:
            int x - o resultado de z calculada na função sigmoide
    '''

    return 1/(1 + e**(-z))

def loss(X, y, w, N):
    '''
    ENTRADA:

            vector  X:  uma matriz de vetores x de features

            vector  y:  um vetor de classificações y^(n) \in {0,1} para cada vetor x 

            vector  w:  vetor de pesos para a reta final

            int     N:  tamanho da matriz X    

    FUNCIONAMENTO:
            o método retornará o valor da perda (binary cross-entry)
            da regressão logística

    SAÍDA:
            valor numérico da função de perda
    '''

    somatorio = 0

    for i in range(N):

        y_hat = theta(X[i] @ w)

        somatorio += y[i] * log(y_hat) + (1 - y[i])*log(1 - y_hat)  

    return (-1/N) * somatorio


def dloss_dw(X, y, w, N):
    '''
    ENTRADA:

            vector  X:  uma matriz de vetores x de features

            vector  y:  um vetor de classificações y^(n) \in {0,1} para cada vetor x 

            vector  w:  vetor de pesos para a reta final

            int     N:  tamanho da matriz X 

    FUNCIONAMENTO:
            o método irá calcular a derivada parcial da função loss()
            para encontrar a variação delta w do vetor de pesos

    SAÍDA:
            float somatorio - resultado do delta wj
    '''

    somatorio = 0

    for i in range(N):
        y_hat = theta(w @ X[i])

        somatorio += (y[i] - y_hat)*X[i]

    return somatorio
    


def main():
    epocas  = int(sys.argv[1])      # numero de iteracoes
    alpha   = float(sys.argv[2])    # learning rate
    epsilon = float(sys.argv[3])    # erro tolerado
    
    ### dados do california housing ###

    X_REAL = dados().data[:, 0]                             # dados de entrada [0,5]
    
    X_REAL = (X_REAL - np.mean(X_REAL)) / np.std(X_REAL)    # normalização dos dados
    X = np.c_[np.ones(X_REAL.shape[0]), X_REAL]             # shape (N, 2)

    Y_REAL = dados().target                                 # variável alvo (preço médio de casa)
    y = (Y_REAL > np.median(Y_REAL)).astype(int)            # binariza os rótulos para regressão logística
    
    ############

    w = np.zeros(X.shape[1])        # vetor de pesos, inicializado com 0's
    N = len(y)


    for epoca in range(epocas):

        if epoca % 50 == 0:
            
            perda_atual = loss(X, y, w, N)
            print(f"Época {epoca}: perda = {perda_atual:.4f}")

        grad = dloss_dw(X, y, w, epocas)
        w   += alpha * grad

        if np.linalg.norm(grad) < epsilon:
            break



if __name__ == "__main__":
    main()
