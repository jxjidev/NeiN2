Controle do Pêndulo Invertido com FIS, Genético-FIS e Neuro-FIS 🤖🎯
Este projeto implementa e compara três sistemas de controle para o problema clássico do pêndulo invertido:

FIS (Sistema de Inferência Fuzzy) tradicional

Genético-FIS: otimização dos parâmetros fuzzy via Algoritmo Genético

Neuro-FIS: controlador treinado com rede neural baseado no FIS

🚀 Como funciona
O sistema simula o controle de um pêndulo invertido sobre um carrinho, buscando mantê-lo estável (ângulo perto de zero) e a posição do carrinho centralizada.

Componentes do projeto:
Simulação dinâmica do pêndulo e carrinho

Controlador fuzzy clássico usando a biblioteca scikit-fuzzy

Otimização genética para melhorar o sistema fuzzy

Treinamento Neuro-FIS com PyTorch para aproximar o controlador fuzzy

Comparação dos três controladores via gráficos e métricas

⚙️ Requisitos
Python 3.8+

Bibliotecas necessárias:
numpy, scikit-fuzzy, matplotlib, torch, deap

Instalação das dependências:
bash
Copy
Edit
pip install numpy scikit-fuzzy matplotlib torch deap
📄 Como executar
Salve o código principal em um arquivo .py.

Execute no terminal:

bash
Copy
Edit
python seu_arquivo.py
O processo realiza:

Treinamento e otimização

Simulações com os 3 controladores

Geração dos gráficos e métricas

Os gráficos serão salvos no arquivo:
comparacao_final_controles_completo.png

📊 Resultados esperados
Métricas exibidas no console:
Controlador	Erro Médio	Variância do Erro	Tempo Médio (s)
FIS	~103.5	~13339	~0.0006
Genético-FIS	~46.8	~2758	~0.0006
Neuro-FIS	~0.21	~0.01	~0.00002

Interpretação:

O Neuro-FIS apresenta erro e variância muito baixos, com alta eficiência computacional.

O Genético-FIS melhora bastante o FIS clássico.

O FIS básico funciona, mas é menos estável e mais impreciso.

📉 Gráficos gerados
O arquivo comparacao_final_controles_completo.png contém:

Força Aplicada ao Longo do Tempo — comparação das forças de controle aplicadas pelos três métodos.

Ângulo do Pêndulo ao Longo do Tempo — mostra estabilidade do pêndulo.

Posição do Carrinho ao Longo do Tempo — mostra deslocamento do carrinho durante controle.

Curva de Convergência do Algoritmo Genético — evolução do erro da melhor solução por geração.

