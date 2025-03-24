import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MaxAbsScaler

print('Carregando Arquivo de teste')
arquivo = np.load('teste5.npy') #mudar o teste
x = arquivo[0]

scale = MaxAbsScaler().fit(arquivo[1])
y = np.ravel(scale.transform(arquivo[1]))

iteracoes = 400
simulacoes = 10  
arquiteturas = [
    (15, 5), 
    (30, 10), 
    (50, 20), 
    (10, 10, 5),
    (20, 10, 5), 
]

erros_finais = {arch: [] for arch in arquiteturas}

for arquitetura in arquiteturas:
    print(f'\nExecutando simulações para arquitetura {arquitetura}')
    
    for i in range(simulacoes):
        print(f'  Executando simulação {i+1}/{simulacoes}')

        regr = MLPRegressor(
            hidden_layer_sizes=arquitetura,
            max_iter=iteracoes,
            activation='tanh',
            solver='adam',
            learning_rate='adaptive',
            n_iter_no_change=iteracoes,
            verbose=False
        )
        
        print('Treinando RNA')
        regr.fit(x, y)
        
        y_est = regr.predict(x)
        
        erro = np.mean((y_est - y) ** 2)
        erros_finais[arquitetura].append(erro)
        print(f'    Erro final da simulação {i+1}: {erro:.5f}')

for arquitetura, erros in erros_finais.items():
    erro_medio = np.mean(erros)
    erro_desvio = np.std(erros)
    print(f'\nArquitetura {arquitetura}:')
    print(f'  Média do erro final: {erro_medio:.5f}')
    print(f'  Desvio padrão do erro final: {erro_desvio:.5f}')

plt.figure(figsize=[14,7])

plt.subplot(1, 3, 1)
plt.title('Função Original')
plt.plot(x, y, color='green')

plt.subplot(1, 3, 2)
plt.title(f'Curva erro (Best Loss: {regr.best_loss_:.5f})')
plt.plot(regr.loss_curve_, color='red')

plt.subplot(1, 3, 3)
plt.title('Função Original x Função aproximada')
plt.plot(x, y, linewidth=1, color='green')
plt.plot(x, y_est, linewidth=2, color='blue')

plt.show()
