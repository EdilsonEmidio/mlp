import math
import random

dados = [(0,0,0),(0,1,1),(1,0,1),(1,1,0)]


class Neuronio:
    
    def __init__(self,pesos) -> None:#os pesos agora são uma lista
        self.pesos = pesos
        self.somatorio = 0
        self.funcao = 0
        self.entradas = []
        self.bies = random.uniform(0,1)
    
    def somar(self, xn):#xn é uma lista dos x, entradas
        somatorio = self.bies
        self.entradas = xn
        
        for i,x in enumerate(xn):
            somatorio += x*self.pesos[i]
        
        self.somatorio = somatorio
      
    def funcaoAtivacao(self, somatorio):
        funcao = 1/(1+ math.exp(-somatorio))
        self.funcao = funcao
        

def criarNeuronio(quantEntrada):
    lista = []
    for n in range(quantEntrada):
        lista.append(random.uniform(0,1))
    
    return Neuronio(lista)

def criarCamada(quantNeuronio,quantEntrada):
    
    camada = []#posicao é so neuronio agora

    for n in range(quantNeuronio):
        camada.append(criarNeuronio(quantEntrada))
    
    return camada


def processar(camadas, batch):
    elementos = [batch[0],batch[1]] #elementos tem 2 entradas x1,x2
    
    ultimaSaida = []
    for i, camada in enumerate(camadas):
        if i==0:#primeira camada
            for neuronio in camada:#camada[1] é o bies
                neuronio.somar(elementos)
                neuronio.funcaoAtivacao(neuronio.somatorio)
                ultimaSaida.append(neuronio.funcao)
        
        else:#todas as outras camadas
            aux = [] #para repassar os ultimos funcaos feitos para a outra lista
            for neuronio in camada:
                neuronio.somar(ultimaSaida)
                neuronio.funcaoAtivacao(neuronio.somatorio)
                aux.append(neuronio.funcao)
            ultimaSaida = aux
                
    y = ultimaSaida[0]
    y2 = 1 if y>0.5 else 0
    
    global erros
    #print("predito ",y, " - real ",batch[2])
    print("valor Y / esperado = ", y2 ," / ", batch[2])
    
    corrigir(camadas, len(camadas)-1, 0 ,(batch[2]-y)*-1)
    if y2 != batch[2]:
        erros = 1
    
    
def corrigir(camadas, quant, posicao, erro):#y é a saida esperada, a real
    lr = 0.01 #learn rate
    neuronio = camadas[quant][posicao]

    derivada = (erro)*neuronio.funcao*(1-neuronio.funcao)
    for j, peso in enumerate(neuronio.pesos):
        if quant>0:
            neuronio.pesos[j] -= lr * derivada*camadas[quant-1][j].funcao
            neuronio.bies -= lr * derivada
            corrigir(camadas, quant-1, j, derivada*peso) #j é o neuronio da outra camada
            
        else:
            neuronio.pesos[j] -= lr * derivada*neuronio.entradas[j]
            neuronio.bies -= lr * derivada
        
        
def sigmoide(somatorio):
    return 1/(1+ math.exp(-somatorio))

def mostrar(camadas):
    for camada in camadas:
        print("bies ",camada[1])
        for neuronio in camada[0]:
            for peso in neuronio.pesos:
                print("pesos", peso)    


def testar(camadas, batch):
    elementos = [batch[0],batch[1]] #elementos tem 2 entradas x1,x2
    
    ultimaSaida = []
    for i, camada in enumerate(camadas):
        if i==0:#primeira camada
            for neuronio in camada:#camada[1] é o bies
                neuronio.somar(elementos)
                neuronio.funcaoAtivacao(neuronio.somatorio)
                ultimaSaida.append(neuronio.funcao)
        
        else:#todas as outras camadas
            aux = [] #para repassar os ultimos funcaos feitos para a outra lista
            for neuronio in camada:
                neuronio.somar(ultimaSaida)
                neuronio.funcaoAtivacao(neuronio.somatorio)
                aux.append(neuronio.funcao)
            ultimaSaida = aux
                
    y = ultimaSaida[0]
    y2 = 1 if y>0.5 else 0
    print(ultimaSaida)
    global erros
    
    print("(",batch[0] ,batch[1],") = ", y2)
    
    
def iniciar(epocas, folds):
    #quantidade neuronios, e quantidade de entrada do neuronio
    camada0 = criarCamada(2,2)
    camada1 = criarCamada(1,2)#tipo assim -> \
   
    camadas = [camada0, camada1]

    for i in range(epocas):
        global erros
        erros = 0
        for batch in folds:
            processar(camadas, batch)
        if erros == 0:
            print("treinado com ",i," epocas!")
            break
        
    
    
    print("terminado")    
    testar(camadas,[1,1])
    testar(camadas,[0,1])
    testar(camadas,[1,0])
    testar(camadas,[0,0])
#epocas é o 1 parametro
iniciar(10000, dados)

