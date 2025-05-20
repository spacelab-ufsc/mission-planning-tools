# link/__main__.py

from .core.link_budget import link_budget

def main():
    lb = link_budget()
    
    # Exemplo de valores para o cálculo do Link Budget
    resultado = lb.calcular()
    
    print(f"Resultado do Link Budget: {resultado:.2f} dB")

if __name__ == "__main__":
    main()
