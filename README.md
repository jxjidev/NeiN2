🛠️ Controle de Pêndulo Invertido com FIS, Genético-Fuzzy e Neuro-Fuzzy



Bem-vindo ao projeto de Controle de Pêndulo Invertido! Este repositório implementa três técnicas de controle para estabilizar um pêndulo invertido: Sistema de Inferência Fuzzy (FIS), Genético-Fuzzy e Neuro-Fuzzy. Nosso objetivo é manter o ângulo do pêndulo (( \theta )) e a posição do carrinho (( x )) próximos de zero ao longo de 5 segundos. 🚀



📖 Sobre o Projeto

O pêndulo invertido é um problema clássico de controle, onde um pêndulo é montado em um carrinho que se move horizontalmente. O desafio é aplicar forças ao carrinho para estabilizar o pêndulo na posição vertical (( \theta \approx 0 )) e manter o carrinho na posição inicial (( x \approx 0 )).

🎯 Objetivos





Implementar um Sistema FIS com regras fuzzy manuais.



Otimizar as regras com um Algoritmo Genético (Genético-Fuzzy).



Treinar uma Rede Neural para controle adaptativo (Neuro-Fuzzy).



Comparar os três sistemas em termos de MSE (( \theta ) e ( x )) e tempo de estabilização.



🛠️ Tecnologias Utilizadas





Python 3.11+



Bibliotecas:





scikit-fuzzy 🧩: Para o sistema fuzzy.



deap 🧬: Para o algoritmo genético.



tensorflow 🧠: Para a rede neural.



matplotlib 📊: Para visualização de gráficos.



numpy 🔢: Para cálculos numéricos.



🚀 Como Configurar e Executar

Siga os passos abaixo para configurar o projeto no Visual Studio Code e executar o script.

1️⃣ Pré-requisitos





Python 3.11+ instalado.



Visual Studio Code com a extensão Python instalada.



Bibliotecas Python necessárias:





scikit-fuzzy



deap



tensorflow



matplotlib



numpy

2️⃣ Configuração do Ambiente





Crie um Ambiente Virtual (Opcional, mas recomendado):

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows



Instale as Dependências: No terminal, execute:

pip install scikit-fuzzy deap tensorflow matplotlib numpy



Configure o VS Code:





Abra o VS Code e instale a extensão Python (da Microsoft).



Abra o arquivo pendulum_control.py (disponível neste repositório).

3️⃣ Executando o Projeto





Abra o Script:





Certifique-se de que pendulum_control.py está no diretório do projeto.



Execute o Script:





Abra o terminal integrado no VS Code (Ctrl+ouView > Terminal`).



Execute:

python pendulum_control.py



Ou use o botão "Run Python File" (triângulo verde no topo).



Acompanhe o Progresso:





O terminal mostrará o progresso, incluindo tempos de execução para cada etapa.



Gráficos comparativos serão exibidos ao final.



📜 Estrutura do Código

O script pendulum_control.py é dividido em quatro seções principais:





Sistema FIS 🧩:





Define variáveis fuzzy, funções de pertinência e regras manuais.



Simula o sistema por 5 segundos usando Runge-Kutta de 4ª ordem.



Sistema Genético-Fuzzy 🧬:





Otimiza as regras fuzzy usando um Algoritmo Genético (AG).



Simula o sistema com as regras otimizadas.



Sistema Neuro-Fuzzy 🧠:





Treina uma rede neural com dados do FIS.



Simula o sistema usando as previsões da rede.



Comparação 📊:





Calcula métricas (MSE e tempo de estabilização).



Gera gráficos comparativos para ( \theta ), ( x ) e força.

⚡ Otimizações Aplicadas





Passo de Tempo: Aumentado para h = 0.01 para acelerar as simulações.



Genético-Fuzzy:





População: 10 indivíduos.



Gerações: 5.



Horizonte de avaliação: 1 segundo.



Neuro-Fuzzy:





Dados de treinamento reduzidos (10x10x5x5 combinações).



Rede neural com 10 épocas e camadas de dropout.



Monitoramento: Tempos de execução são impressos para cada etapa.



📊 Resultados

Após a execução, o script imprimirá métricas de desempenho e exibirá gráficos comparativos.

📈 Métricas de Desempenho

Exemplo de saída no terminal:

Métricas de Desempenho (em 5s):
FIS - MSE Ângulo: 12.34, MSE Posição: 0.56, Tempo de Estabilização: 2.50s
Genético-Fuzzy - MSE Ângulo: 8.90, MSE Posição: 0.45, Tempo de Estabilização: 2.00s
Neuro-Fuzzy - MSE Ângulo: 5.67, MSE Posição: 0.34, Tempo de Estabilização: 1.80s





MSE Ângulo: Erro médio quadrático do ângulo (( \theta )).



MSE Posição: Erro médio quadrático da posição (( x )).



Tempo de Estabilização: Tempo até ( |\theta| < 1^\circ ), ou 5s se não estabilizar.

📉 Gráficos Comparativos

Os gráficos mostram o desempenho dos três sistemas ao longo do tempo:

Comparação do Ângulo (( \theta ))







Descrição: Compara o ângulo do pêndulo (( \theta )) para FIS (azul), Genético-Fuzzy (laranja) e Neuro-Fuzzy (verde). O objetivo é ( \theta \to 0 ).

Comparação da Posição (( x ))







Descrição: Compara a posição do carrinho (( x )) para os três sistemas. O objetivo é ( x \to 0 ).

Comparação da Força de Controle







Descrição: Compara a força de controle aplicada ao carrinho pelos três sistemas.

Nota: Para gerar os gráficos acima, adicione plt.savefig('nome_do_arquivo.png') antes de plt.show() no script e execute novamente. Os arquivos serão salvos no diretório do projeto.



🔧 Possíveis Ajustes

Se os resultados não forem satisfatórios, experimente:





Aumentar o Amortecimento: Aumente b (e.g., para 0.3) se o pêndulo não estabilizar.



Mais Gerações no AG: Aumente ngen (e.g., para 10) para melhorar o Genético-Fuzzy.



Treinamento do Neuro-Fuzzy: Aumente o número de épocas (e.g., para 20) ou adicione mais camadas.



Passo de Tempo: Reduza h (e.g., para 0.005) para maior precisão, mas isso aumentará o tempo de execução.

🐞 Depuração no VS Code
