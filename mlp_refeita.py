import math
import random

dados = [(0,0,0),(0,1,1),(1,0,1),(1,1,0)]


class Neuronio:
    
    def __init__(self,pesos) -> None:#os pesos agora são uma lista
        self.pesos = pesos
        self.somatorio = 0
        self.funcao = 0
        self.entradas = []
    
    def somar(self, bies, xn):#xn é uma lista dos x, entradas
        somatorio = bies
        self.entradas = xn
        
        for i,x in enumerate(xn):
            somatorio += x*self.pesos[i]
        
        self.somatorio = somatorio
      
    def funcaoAtivacao(self, f , somatorio):
        funcao = f(somatorio)
        self.funcao = funcao
        

def criarNeuronio(quantEntrada):
    lista = []
    for n in range(quantEntrada):
        lista.append(random.uniform(0,1))
    
    return Neuronio(lista)

def criarCamada(quantNeuronio,quantEntrada):
    
    bies = random.uniform(0,1)
    camada = [[],bies]#posicao 0 são os neuronios e a 1 é o bias

    for n in range(quantNeuronio):
        camada[0].append(criarNeuronio(quantEntrada))
    
    return camada


def processar(camadas, batch):
    elementos = [batch[0],batch[1]] #elementos tem 2 entradas x1,x2
    
    ultimaSaida = []
    for i, camada in enumerate(camadas):
        if i==0:#primeira camada
            for neuronio in camada[0]:#camada[1] é o bies
                neuronio.somar(camada[1],elementos)
                neuronio.funcaoAtivacao(sigmoide, neuronio.somatorio)
                ultimaSaida.append(neuronio.funcao)
        
        else:#todas as outras camadas
            aux = [] #para repassar os ultimos funcaos feitos para a outra lista
            for neuronio in camada[0]:
                neuronio.somar(camada[1],ultimaSaida)
                neuronio.funcaoAtivacao(sigmoide, neuronio.somatorio)
                aux.append(neuronio.funcao)
            ultimaSaida = aux
    #print("saida ",ultimaSaida)
    
    y = ultimaSaida[0]
    y2 = 1 if y>0.8 else 0
    
    global erros
    
    #print("valor Y / esperado = ", y2 ," / ", batch[2])
    corrigir(camadas, len(camadas)-1, (batch[2]-y)*-1)
    if y2 != batch[2]:
        erros = 1
        
    
    
def corrigir(camadas, quant, erro):#y é a saida esperada, a real
    lr = 0.01 #learn rate
    
    for i, neuronio in enumerate(camadas[quant][0]):
        for j, peso in enumerate(neuronio.pesos):
            derivada = (erro)*neuronio.funcao*(1-neuronio.funcao)
            
            if quant>0:
                neuronio.pesos[j] -= lr * derivada*camadas[quant-1][0][j].funcao                
                camadas[quant][1] -= lr * derivada
                
                corrigir(camadas, quant-1, derivada*neuronio.pesos[j])
            else:
                neuronio.pesos[j] -= lr * derivada*neuronio.entradas[j]
                camadas[quant][1] -= lr * derivada
        
    
def hiperbolica(somatorio):
    return (math.exp(somatorio) - math.exp(-somatorio)) / (math.exp(somatorio) + math.exp(-somatorio))

def derivadaHiperbolica():
    pass
    
def sigmoide(somatorio):
    return 1/(1+ math.exp(-somatorio))

def derivadaSigmoide(sigmoide):
    return sigmoide*(1- sigmoide)


def mostrar(camadas):
    for camada in camadas:
        print("bies ",camada[1])
        for neuronio in camada[0]:
            for peso in neuronio.pesos:
                print("pesos", peso)
        
    
def iniciar(epocas, folds):
    #quantidade neuronios, e quantidade de entrada do neuronio
    camada0 = criarCamada(2,2)
    camada1 = criarCamada(1,2)#tipo assim -> \

    camadas = [camada0, camada1]

    mostrar(camadas)
    #print("acabado-------------")
    for i in range(epocas):
        global erros
        erros = 0
        for batch in folds:
            processar(camadas, batch)
        if erros == 0:
            print("treinado com ",i," epocas!")
            break
    #mostrar(camadas)
    print("terminado")    

#epocas é o 1 parametro
iniciar(100, dados)