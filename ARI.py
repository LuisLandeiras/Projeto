def ARI(File) -> float:
    with open(File, "r", encoding='utf-8', errors='ignore') as file:
        sentence = file.read().replace("\n", " ")
    
    words = len(sentence.split())
    sentences = sentence.count('.') + sentence.count('!') + sentence.count('?') #Conta o número de frases tendo em conta as pontuações
    characters = len(sentence.replace(" ", "")) #Retira os espaços para contar o número de letras usadas
    
    ari = 4.71 * (characters/words) + 0.5 * (words/sentences) - 21.43 #Formula usada para calcular o ARI
    return ari

print(ARI("The_Mother.txt"))