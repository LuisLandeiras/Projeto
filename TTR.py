import string, random

def TTRMedio(File) -> float:
    with open(File, "r", encoding='utf-8', errors='ignore') as file:
        sentence = file.read().replace("\n", " ")
        sentence = sentence.translate(str.maketrans('','', string.punctuation)) #Limpar a pontuação do texto
    
    Text = sentence.split() #Colocar o texto numa lista

    soma = 0
    for _ in range(1000): #1000 samples são estudadas para o resultado final
        Sample = random.sample(Text,500) #Escolhe 100 palavras random do texto
        ttr = len(set(Sample))/len(Sample) #Divisão entre a sample limpa(Palavras não repetidas) e o tamanho total da sample
        soma += ttr
    return soma/1000

print(f"TTR Biden: {TTRMedio('biden.txt'):.5f}")
print(f"TTR Trump: {TTRMedio('trump.txt'):.5f}")
print(f"TTR Obama: {TTRMedio('obama.txt'):.5f}")

print(f"TTR The_Mother: {TTRMedio('The_Mother.txt'):.5f}")
print(f"TTR Men_Withour_Women: {TTRMedio('Men_Without_Women.txt'):.5f}")