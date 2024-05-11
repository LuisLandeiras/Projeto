import spacy, random, csv, Tree, TCM, threading, RM, SA, GC

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 1600000

csv.field_size_limit(2147483647)

def Amostras(Texto):
    doc = nlp(Texto)

    Palavras = [token.text for token in doc if token.is_alpha] # Separa cada token, guardando só palavras 
    
    Sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.split()) >= 4]
    
    Frases = random.sample(Sentences,40)
    
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
            'WordLength',
            'Classification'
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
                ResultadosA['WordLength'][0],
                2
            ])
    return

def CsvFilter(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 7:
                text = ','.join(parts[6:])
                if len(text.split()) > 5000:
                    f_out.write(','.join(parts) + '\n')

def CsvTier(input_csv, output_csv):
    with open(input_csv, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        headers = reader.fieldnames
        filtered_rows = [row for row in reader if 33 <= int(row['age']) <= 45]

    with open(output_csv, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(filtered_rows)

def Csv_Ave(csv_file):
    total = 0
    count = 0

    with open(csv_file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        
        for row in reader:
            total += float(row[14])
            count += 1
            
        print(total/count)
            



#Csv_Ave('Tier1Algo.csv')
#print("----------------")
#Csv_Ave('Tier2Algo.csv')
#print("----------------")
#Csv_Ave('Tier3Algo.csv')
#print("----------------")
#Csv_Ave('Tier4Algo.csv')
#print("----------------")

#CsvTier("Tier.csv", "Tier3.csv")
#CsvFilter("blogtext.csv", "Tier.csv")
#Txt_Csv("Textos/","teste.csv")
#Csv_Csv("TBons.csv", "Texto.csv")
#CsvAlgo("Tier2.csv", "Tier2Algo.csv")

#Tier1 = 13-22; Tier2 = 23-32; Tier3 = 33-45
