# Controle do Pêndulo Invertido com FIS, Genético-FIS e Neuro-FIS 🤖🎯

Este projeto implementa e compara três sistemas de controle para o problema clássico do pêndulo invertido:

- **FIS (Sistema de Inferência Fuzzy)** tradicional  
- **Genético-FIS**: otimização dos parâmetros fuzzy via Algoritmo Genético  
- **Neuro-FIS**: controlador treinado com rede neural baseado no FIS  

---

## 🚀 Como funciona

O sistema simula o controle de um pêndulo invertido sobre um carrinho, buscando mantê-lo estável (ângulo perto de zero) e a posição do carrinho centralizada.

### Componentes do projeto:

- Simulação dinâmica do pêndulo e carrinho  
- Controlador fuzzy clássico usando a biblioteca `scikit-fuzzy`  
- Otimização genética para melhorar o sistema fuzzy  
- Treinamento Neuro-FIS com PyTorch para aproximar o controlador fuzzy  
- Comparação dos três controladores via gráficos e métricas  

---

## ⚙️ Requisitos

- Python 3.8+  
- Bibliotecas necessárias:  
  `numpy`, `scikit-fuzzy`, `matplotlib`, `torch`, `deap`  

### Instalação das dependências:

```bash
pip install numpy scikit-fuzzy matplotlib torch deap
