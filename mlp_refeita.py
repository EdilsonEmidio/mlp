import math
import random
    
dados = [(0,0,0),(0,1,0),(1,0,0),(1,1,1)]
euler = 2.71828

class Neuronio:
    
    def __init__(self,pesos) -> None:#os pesos agora são uma lista
        self.pesos = pesos
        self.somatorio = 0
        self.sigmoide = 0
        self.entradas = []
    
    def somar(self, bies, xn):#xn é uma lista dos x, entradas
        somatorio = bies
        self.entradas = xn
        
        for i,x in enumerate(xn):
            somatorio += x*self.pesos[i]
        
        self.somatorio = somatorio
      
    def funcaoAtivacao(self, somatorio):
        sigmoide = 1/(1+ math.exp(-somatorio))
        self.sigmoide = sigmoide
        

def criarNeuronio(quantEntrada):
    lista = []
    for n in range(quantEntrada):
        lista.append(random.uniform(-1,1))
    
    return Neuronio(lista)

def criarCamada(quantNeuronio,quantEntrada):
    bies = random.uniform(-1,1)
    camada = [[],bies]#posicao 0 são os neuronios e a 1 é o bias

    for n in range(quantNeuronio):
        camada[0].append(criarNeuronio(quantEntrada))
    
    return camada

def mostrarRede(camadas):
    for i,camada in enumerate(camadas):
        print("camada ",i)
        for j,neuronio in enumerate(camada[0]):
            print("neuronio ",j,neuronio.sigmoide)

def processar(camadas, batch):
    elementos = [batch[0],batch[1]] #elementos tem 2 entradas x1,x2
    
    ultimaSaida = []
    for i, camada in enumerate(camadas):
        if i==0:#primeira camada
            for neuronio in camada[0]:#camada[1] é o bies
                neuronio.somar(camada[1],elementos)
                neuronio.funcaoAtivacao(neuronio.somatorio)
                ultimaSaida.append(neuronio.sigmoide)
        
        else:#todas as outras camadas
            aux = [] #para repassar os ultimos sigmoides feitos para a outra lista
            for neuronio in camada[0]:
                neuronio.somar(camada[1],ultimaSaida)
                neuronio.funcaoAtivacao(neuronio.somatorio)
                aux.append(neuronio.sigmoide)
            ultimaSaida = aux
    print("saida ",ultimaSaida)
    
    y = ultimaSaida[0]
    y2 = 1 if y>=0 else 0
    
    #print("valor Y / esperado = ", y2 ," / ", batch[2])
    #mostrarRede(camadas)
    corrigir(camadas, len(camadas)-1, batch[2])
    
    
def corrigir(camadas, quant, erro):#y é a saida esperada, a real
    lr = 0.5 #learn rate
    
    for i, neuronio in enumerate(camadas[quant][0]):
        for j, peso in enumerate(neuronio.pesos):
            derivada = (erro)*neuronio.sigmoide*(1-neuronio.sigmoide)
            if quant>0:
                neuronio.pesos[j] -= lr * derivada*camadas[quant-1][0][j].sigmoide
                camadas[quant][1] -= lr * derivada
                
                corrigir(camadas, quant-1, derivada*neuronio.pesos[j])
            else:
                neuronio.pesos[j] -= lr * derivada*neuronio.entradas[j]
                camadas[quant][1] -= lr * derivada
        
        
def iniciar(epocas, folds):
    #quantidade neuronios, e quantidade de entrada do neuronio
    camada0 = criarCamada(2,2)
    camada1 = criarCamada(1,2)#tipo assim -> \
    #camada2 = criarCamada(1,2)
  

    camadas = [camada0, camada1]
    
    for i in range(epocas):
        for batch in folds:
            processar(camadas, batch)
    #processar(camadas, folds[0])
    print("tudo feito")    
    
#epocas é o 1 parametro
iniciar(100, dados)