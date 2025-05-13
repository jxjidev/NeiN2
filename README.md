#Controle de Pêndulo Invertido com FIS, Genético-Fuzzy e Neuro-Fuzzy

Descrição do Projeto

Este projeto implementa o controle de um pêndulo invertido utilizando três abordagens: Sistema de Inferência Fuzzy (FIS), Genético-Fuzzy e Neuro-Fuzzy. O objetivo é estabilizar o pêndulo (( \theta \approx 0 )) e o carrinho (( x \approx 0 )) ao longo de 5 segundos, aplicando forças de controle apropriadas. O projeto é implementado em Python e otimizado para execução no Visual Studio Code (VS Code), embora também possa ser adaptado para Google Colab.

Estrutura do Projeto





pendulum_control.py: Script principal contendo todo o código para o FIS, Genético-Fuzzy, Neuro-Fuzzy e comparação.



README.md: Este documento com instruções e resultados.

Pré-requisitos

Para executar o projeto, você precisará de:





Python 3.11 ou superior instalado.



Visual Studio Code com a extensão Python instalada.



Bibliotecas Python necessárias:





scikit-fuzzy



deap



tensorflow



matplotlib



numpy

Instalação





Instale o Python: Baixe e instale o Python do site oficial. Certifique-se de adicionar o Python ao PATH.



Instale o Visual Studio Code: Baixe e instale o VS Code do site oficial.



Instale a Extensão Python no VS Code:





Abra o VS Code.



Vá para a aba de extensões (Ctrl+Shift+X ou Cmd+Shift+X no Mac).



Procure por "Python" e instale a extensão da Microsoft.



Crie um Ambiente Virtual (Recomendado):

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows



Instale as Bibliotecas: No terminal (com o ambiente virtual ativado, se aplicável), execute:

pip install scikit-fuzzy deap tensorflow matplotlib numpy

Configuração do Projeto





Crie o Arquivo do Script:





No VS Code, crie um novo arquivo chamado pendulum_control.py.



Copie e cole o código fornecido no script principal (disponível no arquivo pendulum_control.py ou na resposta anterior).



Verifique as Dependências:





Certifique-se de que todas as bibliotecas estão instaladas corretamente. Se houver erros ao executar o script, tente reinstalar as bibliotecas ou verificar a versão do Python.

Como Executar





Abra o Projeto no VS Code:





Abra o VS Code e carregue o arquivo pendulum_control.py.



Execute o Script:





Abra o terminal integrado no VS Code (Ctrl+` ou View > Terminal).



Certifique-se de que o ambiente virtual está ativado (se criado).



Execute o script:

python pendulum_control.py



Alternativamente, use o botão "Run Python File" (triângulo verde na barra superior do VS Code).



Acompanhe a Execução:





O script imprimirá o progresso no terminal, incluindo tempos de execução para cada etapa (FIS, Genético-Fuzzy, Neuro-Fuzzy).



Ao final, gráficos comparativos serão exibidos em uma janela do Matplotlib.

Estrutura do Código

O script pendulum_control.py é organizado em seções principais:





Sistema FIS:





Define variáveis fuzzy, funções de pertinência e regras manuais.



Simula o sistema por 5 segundos usando o método Runge-Kutta de 4ª ordem.



Sistema Genético-Fuzzy:





Usa um Algoritmo Genético (AG) para otimizar as regras fuzzy.



Simula o sistema com as regras otimizadas.



Sistema Neuro-Fuzzy:





Treina uma rede neural com dados gerados pelo FIS.



Simula o sistema usando as previsões da rede neural.



Comparação:





Calcula métricas (MSE do ângulo, MSE da posição, tempo de estabilização).



Gera gráficos comparativos para ( \theta ), ( x ) e força.

Otimizações Aplicadas





Passo de Tempo: Aumentado para h = 0.01 para acelerar simulações.



Genético-Fuzzy:





População reduzida para 10 indivíduos.



Gerações reduzidas para 5.



Horizonte de avaliação reduzido para 1 segundo durante a otimização.



Neuro-Fuzzy:





Dados de treinamento reduzidos para 10x10x5x5 combinações.



Rede neural simplificada com 10 épocas.



Monitoramento de Tempo: Incluído para diagnosticar gargalos.

Resultados

Métricas de Desempenho

Após executar o script, as métricas de desempenho serão impressas no console. Um exemplo de saída seria:

Métricas de Desempenho (em 5s):
FIS - MSE Ângulo: 12.34, MSE Posição: 0.56, Tempo de Estabilização: 2.50s
Genético-Fuzzy - MSE Ângulo: 8.90, MSE Posição: 0.45, Tempo de Estabilização: 2.00s
Neuro-Fuzzy - MSE Ângulo: 5.67, MSE Posição: 0.34, Tempo de Estabilização: 1.80s





MSE Ângulo: Erro médio quadrático do ângulo (( \theta )).



MSE Posição: Erro médio quadrático da posição do carrinho (( x )).



Tempo de Estabilização: Tempo até ( |\theta| < 1^\circ ), ou 5s se não estabilizar.

Gráficos Comparativos

Os gráficos gerados ao final do script mostram a comparação entre os três sistemas:

Comparação do Ângulo (( \theta ))







Descrição: Este gráfico mostra o ângulo do pêndulo (( \theta )) ao longo do tempo para FIS (azul), Genético-Fuzzy (laranja) e Neuro-Fuzzy (verde). O objetivo é que ( \theta \to 0 ).

Comparação da Posição (( x ))







Descrição: Este gráfico mostra a posição do carrinho (( x )) ao longo do tempo para os três sistemas. O objetivo é que ( x \to 0 ).

Comparação da Força de Controle







Descrição: Este gráfico mostra a força de controle aplicada ao carrinho ao longo do tempo, comparando os três sistemas.

Nota: Os arquivos comparison_theta.png, comparison_x.png e comparison_force.png não estão incluídos aqui, mas seriam gerados automaticamente se você modificasse o script para salvar os gráficos com plt.savefig() antes de plt.show(). Por exemplo:

plt.savefig('comparison_theta.png')
plt.show()

Interpretação dos Resultados





FIS: O sistema base, com regras manuais, deve estabilizar o pêndulo, mas pode ter oscilações ou erros maiores.



Genético-Fuzzy: Deve melhorar o desempenho do FIS, com menor MSE e tempo de estabilização, devido à otimização das regras.



Neuro-Fuzzy: Geralmente oferece o melhor desempenho, com controle adaptativo baseado no treinamento da rede neural.

Se os gráficos mostrarem divergência (( \theta ) ou ( x ) crescendo sem controle), ajuste o coeficiente de amortecimento (b) ou reveja as regras fuzzy.

Possíveis Ajustes





Aumentar o Amortecimento: Se o pêndulo não estabilizar, aumente b (e.g., para 0.3).



Mais Gerações no AG: Se o Genético-Fuzzy não melhorar o FIS, aumente ngen (e.g., para 10).



Treinamento do Neuro-Fuzzy: Aumente o número de épocas (e.g., para 20) ou adicione mais camadas à rede neural.



Passo de Tempo: Se precisar de mais precisão, reduza h (e.g., para 0.005), mas isso aumentará o tempo de execução.
