import  sys
import  os
import  matplotlib.pyplot as plt
import  numpy as np
from    sklearn.datasets import fetch_california_housing as dados

e = np.exp(1)

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

    X_REAL = dados().data[:, [0,5]]                         # dados de entrada [0,5]
    
    # normalização dos dados
    X_REAL = (X_REAL - np.mean(X_REAL, axis=0)) / np.std(X_REAL, axis=0)
    
    X = np.c_[np.ones(X_REAL.shape[0]), X_REAL]             # shape (N, 3)

    Y_REAL = dados().target                                 # variável alvo (preço médio de casa)
    y = (Y_REAL > np.median(Y_REAL)).astype(int)            # binariza os rótulos para regressão logística
    
    ############

    w = np.zeros(X.shape[1])        # vetor de pesos, inicializado com 0's
    N = len(y)

    perdas = []


    for epoca in range(epocas):

        if epoca % 50 == 0:
            
            perda_atual = loss(X, y, w, N)
            perdas.append(perda_atual)

            print(f"Época {epoca}: perda = {perda_atual}")

        grad = dloss_dw(X, y, w, N)
        w   += alpha * grad

        if np.linalg.norm(grad) < epsilon:
            break

    ############## PLOTAGEM ################
    fig, ax = plt.subplots(figsize=(12,8))

    # Scatter com as duas features: X[:,1] (feature 1), X[:,2] (feature 2)
    ax.scatter(X[y==0,1], X[y==0,2], color='red', label='Classe 0 (negativa)', alpha=0.5, s=10)
    ax.scatter(X[y==1,1], X[y==1,2], color='blue', label='Classe 1 (positiva)', alpha=0.5, s=10)

    # Gera reta de separação: w0 + w1*x1 + w2*x2 = 0
    # ⇒ x2 = -(w0 + w1*x1)/w2

    x1_vals = np.linspace(X[:,1].min(), X[:,1].max(), 300)
    
    if w[2] != 0:
        x2_vals = -(w[0] + w[1]*x1_vals) / w[2]
        ax.plot(x1_vals, x2_vals, color='green', linestyle='--', label='Fronteira de decisão')
    
    else:
        ax.axvline(x=-w[0]/w[1], color='green', linestyle='--', label='Fronteira de decisão')

    # cálculo da acurácia
    y_pred = (theta(X @ w) >= 0.5).astype(int)
    acuracia = np.mean(y_pred == y)
    print(f"Acurácia final: {acuracia*100:.2f}%")

    ax.set_xlabel("Feature 1 (normalizada)")
    ax.set_ylabel("Feature 2 (normalizada)")
    ax.set_title(f"Fronteira de decisão da Regressão Logística com duas variáveis do California Housing \n {epocas} Épocas, epsilon = {epsilon} - Acurácia final: {acuracia}")
    ax.grid(True)
    ax.legend()
    

    diretorio = "Plotagens"
    if not os.path.exists(diretorio):
        os.makedirs(diretorio)

    # Gera um nome de arquivo que não sobrescreva
    base = f"reglog-epc{epocas}-eps{epsilon}"
    extensao = ".png"
    contador = 1
    caminho_arquivo = os.path.join(diretorio, f"{base}{extensao}")

    # Incrementa nome até encontrar um nome livre
    while os.path.exists(caminho_arquivo):
        caminho_arquivo = os.path.join(diretorio, f"{base}_{contador}{extensao}")
        contador += 1

    # Salva o gráfico
    plt.savefig(caminho_arquivo, format='png')
    print(f"Gráfico salvo como: {caminho_arquivo}")

    plt.show()

    #########################################

if __name__ == "__main__":
    main()
