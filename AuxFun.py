import spacy, random, csv, Tree, TCM, threading, RM, SA_NLTK, os, SA_Spacy, SA_NLTK
import pandas as pd
import numpy as np

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 1600000

csv.field_size_limit(2147483647)

def File(file):
    with open(file, "r", encoding='utf-8') as file:
        content = file.read()
    return content

def Amostras(Texto):
    doc = nlp(Texto)
    
    Sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.split()) >= 4]
    
    # Lista onde é guardada 60 frases
    Frases = random.sample(Sentences,60)
    
    return Frases # Escolhe de forma random 60 frases de um texto

def Resultados(Samples):
    ResultadosA = {}
            
    def TSMOGA(): ResultadosA['Smog'] = RM.SMOGA(Samples)
    def TColemanA(): ResultadosA['Coleman'] = RM.ColemanA(Samples)
    def TGradeA(): ResultadosA['Grade'] = RM.FleschGradeA(Samples)
    def TReadingA(): ResultadosA['Reading'] = RM.FleschReadingA(Samples)
    def TARIA(): ResultadosA['ARI'] = RM.ARIA(Samples)
    
    #TCM Amostras
    def TSentenceLengthA(): ResultadosA['SentenceLength'] = TCM.SentenceLengthA(Samples)
    def TWordLengthA(): ResultadosA['WordLength'] = TCM.WordLengthA(Samples)
    def TLexicalDensityA(): ResultadosA['LexicalDensity'] = TCM.LexicalDensityA(Samples)
    def TLexicalDiversityA(): ResultadosA['LexicalDiversity'] = TCM.LexicalDiversityA(Samples)
    
    #Tree Amostras
    def TTreeA(): ResultadosA['Tree'] = Tree.DepthAveA(Samples)
    
    #SA Amostras
    def TSentimentA(): ResultadosA['Sentiment'] = SA_NLTK.SentimentA(Samples)
    
    #GC
    #def TGrammar(): ResultadosA['Grammar'] = GC.Grammar(Samples[1])
    
    # Create threads
    threads = [
        threading.Thread(target=TSentenceLengthA),
        threading.Thread(target=TWordLengthA),
        threading.Thread(target=TLexicalDensityA),
        threading.Thread(target=TLexicalDiversityA),
        threading.Thread(target=TTreeA),
        threading.Thread(target=TSentimentA),
        #threading.Thread(target=TGrammar),
        threading.Thread(target=TGradeA),
        threading.Thread(target=TSMOGA),
        threading.Thread(target=TColemanA),
        threading.Thread(target=TReadingA),
        threading.Thread(target=TARIA),
    ]
    # Start threads
    for thread in threads: thread.start()
        
    # Wait for all threads to finish
    for thread in threads: thread.join()

    return ResultadosA

def ResultadosC(Samples):
    ResultadosA = {}
            
    def Read(): ResultadosA['Read'] = RM.Read(Samples)
     
    #TCM Amostras
    def TSentenceLengthA(): ResultadosA['SentenceLength'] = TCM.SentenceLength(Samples)
    def TWordLengthA(): ResultadosA['WordLength'] = TCM.WordLength(Samples)
    def TLexicalDensityA(): ResultadosA['LexicalDensity'] = TCM.LexicalDensity(Samples)
    def TLexicalDiversityA(): ResultadosA['LexicalDiversity'] = TCM.LexicalDiversity(Samples)
    
    #Tree Amostras
    def TTreeA(): ResultadosA['Tree'] = Tree.DepthAve(Samples)
    
    #SA Amostras
    def TSentimentA(): ResultadosA['Sentiment'] = SA_Spacy.Sentiment(Samples)
    
    #GC
    #def TGrammar(): ResultadosA['Grammar'] = GC.Grammar(Samples)
    
    # Create threads
    threads = [
        threading.Thread(target=TSentenceLengthA),
        threading.Thread(target=TWordLengthA),
        threading.Thread(target=TLexicalDensityA),
        threading.Thread(target=TLexicalDiversityA),
        threading.Thread(target=TTreeA),
        threading.Thread(target=TSentimentA),
        #threading.Thread(target=TGrammar),
        threading.Thread(target=Read),
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
            text = row['ClassificationS']
            filtered_texts.append(text)

    with open(output_csv, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['T'])

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
            #'Grammar', 
            'LexicalDensity',
            'LexicalDiversity',
            'Reading',
            'SentenceLength',
            'SentimentNeg',
            'SentimentNeu',
            'SentimentPos',
            'Compound',
            'Smog',
            'Tree',
            'WordLength',
            'Classification',
            'ClassificationS'
            ])

        for Texto in filtered_texts:
            Samples = Amostras(Texto)
            ResultadosA = Resultados(Samples)
            
            writer.writerow([
                Texto, 
                ResultadosA['ARI'][0], 
                ResultadosA['Coleman'][0], 
                ResultadosA['Grade'][0], 
                #ResultadosA['Grammar'][0], 
                ResultadosA['LexicalDensity'][0],
                ResultadosA['LexicalDiversity'][0],
                ResultadosA['Reading'][0],
                ResultadosA['SentenceLength'][0],
                ResultadosA['Sentiment'][0],
                ResultadosA['Sentiment'][1],
                ResultadosA['Sentiment'][2],
                ResultadosA['Sentiment'][3],
                ResultadosA['Smog'][0],
                ResultadosA['Tree'][0],
                ResultadosA['WordLength'][0],
                0,
                0
            ])
    return

#Filtra os textos com mais de 5000 palavras
def CsvFilter(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 7:
                text = ','.join(parts[6:])
                if len(text.split()) > 4000:
                    f_out.write(','.join(parts) + '\n')

#Filtra os textos por idades
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
            total += float(row[13])
            count += 1
            
    return round(total/count,3)

#Copia uma diretoria de ficheiros txt para csv     
def txt_to_csv(input_dir, output_file):
    # Open CSV file for writing
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Iterate through each file in the input directory
        for filename in os.listdir(input_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(input_dir, filename)
                
                # Read contents of the file
                with open(file_path, 'r', encoding='utf-8') as txtfile:
                    content = txtfile.read().replace('\n', ' ')  # Replace newlines with space
                    writer.writerow([content])

def Ave(File):
    df = pd.read_csv(File)
    
    dfs_num = df.drop(columns=['Text','ARI','Coleman','Grade','LexicalDensity','LexicalDiversity','Reading','SentenceLength','Smog','Tree','WordLength','Classification','ClassificationS'])
    
    df_num = df.drop(columns=['Text','Classification','SentimentNeg','SentimentNeu','SentimentPos','ClassificationS','Compound'])
    df_num = df_num.apply(pd.to_numeric, errors='coerce')
    df['Average'] = df_num.mean(axis=1).round(2)

    conditions = [
        df['Average'] < 13,
        (df['Average'] >= 13) & (df['Average'] < 13.50),
        (df['Average'] >= 13.50) & (df['Average'] < 14),
        df['Average'] >= 14
    ]
    choices = [0, 1, 2, 3]
    
    conditionss = [
        (dfs_num['Compound'] < -0.2) & (dfs_num['SentimentNeg'] > dfs_num['SentimentPos']),
        (dfs_num['Compound'] >= -0.2) & (dfs_num['Compound'] <= 0.2),
        (dfs_num['Compound'] > 0.2) & (dfs_num['SentimentPos'] > dfs_num['SentimentNeg'])
    ]
    
    #conditionss = [
    #    (dfs_num['Compound'] < -0.2),
    #    (dfs_num['Compound'] >= -0.2) & (dfs_num['Compound'] <= 0.2),
    #    (dfs_num['Compound'] > 0.2)
    #]
    #0: Negativo, 1:Neutro, 2:Positivo
    choicess = [0, 1, 2]
    
    df['Classification'] = np.select(conditions, choices)
    df['ClassificationS'] = np.select(conditionss, choicess)
    df.to_csv('B.csv', index=False)

#Dropa uma coluna especifica
def CSV():
    df = pd.read_csv('B.csv')
    df = df.drop(columns=['Average'])
    df.to_csv('DataV4_2_NLTK.csv', index=False)   

def Count(File):
    df = pd.read_csv(File)

    t0_count = df[df['T'] < 13].shape[0]
    t1_count = df[(df['T'] >= 13) & (df['T'] < 13.50)].shape[0]
    t2_count = df[(df['T'] >= 13.50) & (df['T'] < 14)].shape[0]
    t3_count = df[df['T'] >= 14].shape[0]

    print(f"t0: {t0_count}, t1: {t1_count}, t2: {t2_count}, t3: {t3_count}")

#CsvAlgo("TextosV3.csv", "TextosAlgoV4.csv")
#Ave("TextosAlgoV4.csv")
#CSV()

#Csv_Csv("B.csv", "T1.csv")
#Count("T.csv")
#print(Csv_Ave("DataV4.csv"))

for files in os.listdir("Textos/"):
    file = File("Textos/" + files)
    amostra = Amostras(file)
    print("\n\n\n"+files)
    print(Resultados(amostra))
    #print("Spacy:", SA_Spacy.Sentiment(file))
    #print("Spacy A:", SA_Spacy.SentimentA(amostra))
    #print("----------------------------------------")
    #print("NLTK:", SA_NLTK.Sentiment(file))
    #print("NLTK A:", SA_NLTK.SentimentA(amostra))





