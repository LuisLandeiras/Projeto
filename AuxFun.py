import spacy, random, csv, Tree, TCM, threading, RM, SA, GC, sys

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 1600000

#maxInt = sys.maxsize
#while True:
#    try:
#        csv.field_size_limit(maxInt)
#        break
#    except OverflowError:
#        maxInt = int(maxInt/10)

def File(file):
    with open(file, 'r', encoding='utf-8') as file:
        Texto = file.read()
    return Texto

def Amostras(Texto):
    doc = nlp(Texto)

    Palavras = [token.text for token in doc if token.is_alpha] # Separa cada token, guardando só palavras 
    
    Sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.split()) >= 4]
    
    Frases = random.sample(Sentences,20)
    
    #Lista onde é guardada 100 repetições com amostras de 100 palavras da lista Palavras
    Samples = []
    for _ in range(100):
        Sample = random.sample(Palavras,100)
        Samples.append(Sample)
    
    return Samples, Frases # [0] Escolhe de forma random 100 amostras de 100 palavras de uma lista tokenizada; [1] Escolhe de forma random 10 frases de um texto

def Resultados(Samples):
    ResultadosA = {}
            
    def TSMOGA(): ResultadosA['Smog'] = RM.SMOGA(Samples[1])
    def TColemanA(): ResultadosA['Coleman'] = RM.ColemanA(Samples[1])
    def TGradeA(): ResultadosA['Grade'] = RM.FleschGradeA(Samples[1])
    def TReadingA(): ResultadosA['Reading'] = RM.FleschReadingA(Samples[1])
    def TARIA(): ResultadosA['ARI'] = RM.ARIA(Samples[1])
    
    #TCM Amostras
    def TSentenceLengthA(): ResultadosA['SentenceLength'] = TCM.SentenceLengthA(Samples[1])
    def TWordLengthA(): ResultadosA['WordLength'] = TCM.WordLengthA(Samples[0])
    def TLexicalDensityA(): ResultadosA['LexicalDensity'] = TCM.LexicalDensityA(Samples[0])
    def TLexicalDiversityA(): ResultadosA['LexicalDiversity'] = TCM.LexicalDiversityA(Samples[0])
    
    #Tree Amostras
    def TTreeA(): ResultadosA['Tree'] = Tree.DepthAveA(Samples[1])
    
    #SA Amostras
    def TSentimentA(): ResultadosA['Sentiment'] = SA.SentimentA(Samples[1])
    
    #GC
    def TGrammar(): ResultadosA['Grammar'] = GC.Grammar(Samples[1])
    
    # Create threads
    threads = [
        threading.Thread(target=TSentenceLengthA),
        threading.Thread(target=TWordLengthA),
        threading.Thread(target=TLexicalDensityA),
        threading.Thread(target=TLexicalDiversityA),
        threading.Thread(target=TTreeA),
        threading.Thread(target=TSentimentA),
        threading.Thread(target=TGrammar),
        threading.Thread(target=TGradeA),
        threading.Thread(target=TSMOGA),
        threading.Thread(target=TColemanA),
        threading.Thread(target=TReadingA),
        threading.Thread(target=TARIA),
    ]
    # Start threads
    for thread in threads: thread.daemon = True
    for thread in threads: thread.start()
        
    # Wait for all threads to finish
    for thread in threads: thread.join()

    return ResultadosA

# Copiar colunas de um csv para outro
def Csv_Csv(input_csv, output_csv):
    filtered_texts = []

    with open(input_csv, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            text = row['Text']
            filtered_texts.append(text)

    with open(output_csv, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Text'])

        for text in filtered_texts:
            writer.writerow([text])

# Calcula os algoritmos para cada texto no csv
def CsvAlgo(input_csv, output_csv):
    filtered_texts = []

    with open(input_csv, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            text = row['Text']
            filtered_texts.append(text)

    with open(output_csv, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Text',
            'ARI', 
            'Coleman', 
            'Grade', 
            'Grammar', 
            'LexicalDensity',
            'LexicalDiversity',
            'Reading',
            'SentenceLength',
            'SentimentNeg',
            'SentimentNeu',
            'SentimentPos',
            'Smog',
            'Tree',
            'WordLength'
            ])

        for Texto in filtered_texts:
            Samples = Amostras(Texto)
            ResultadosA = Resultados(Samples)
            
            writer.writerow([
                Texto, 
                ResultadosA['ARI'][0], 
                ResultadosA['Coleman'][0], 
                ResultadosA['Grade'][0], 
                ResultadosA['Grammar'][0], 
                ResultadosA['LexicalDensity'][0],
                ResultadosA['LexicalDiversity'][0],
                ResultadosA['Reading'][0],
                ResultadosA['SentenceLength'][0],
                ResultadosA['Sentiment'][0],
                ResultadosA['Sentiment'][1],
                ResultadosA['Sentiment'][2],
                ResultadosA['Smog'][0],
                ResultadosA['Tree'][0],
                ResultadosA['WordLength'][0]
            ])
    return

#Txt_Csv("Textos/","teste.csv")
#Csv_Csv("TBons.csv", "Texto.csv")
#CsvAlgo("Resto.csv", "TextoAlgo2.csv")

#Texto = File("Textos/trump.txt")
#
#print(Resultados(Amostras(Texto)))
#
#print(TCM.LexicalDensity(Texto))
#print(TCM.LexicalDiversity(Texto))
#print(TCM.SentenceLength(Texto))
#print(TCM.WordLength(Texto))
#print(RM.Read(Texto))
##print(GC.Grammar(File("Textos/trump.txt")))
#print(Tree.DepthAve(Texto))
#print(SA.Sentiment(Texto))
