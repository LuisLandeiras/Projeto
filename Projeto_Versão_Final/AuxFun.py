import spacy, random, csv, Tree, TCM, threading, RM, os, SA_Spacy, WR, TR, Emotions
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
      
    #RM Amostras   
    def TSMOGA(): ResultadosA['Smog'] = RM.SMOGA(Samples)
    def TReadA():
        Read = RM.ReadA(Samples) 
        ResultadosA['Coleman'] = Read[2]
        ResultadosA['Grade'] = Read[0]
        ResultadosA['ARI'] = Read[1]
        ResultadosA['Lix'] = Read[5]
        ResultadosA['Rix'] = Read[6]
        ResultadosA['GFog'] = Read[4]
    
    #TCM Amostras
    def TSentenceLengthA(): ResultadosA['SentenceLength'] = TCM.SentenceLengthA(Samples)
    def TWordLengthA(): ResultadosA['WordLength'] = TCM.WordLengthA(Samples)
    def TLexicalDensityA(): ResultadosA['LexicalDensity'] = TCM.LexicalDensityA(Samples)
    def TLexicalDiversityA(): ResultadosA['LexicalDiversity'] = TCM.LexicalDiversityA(Samples)
    def TSyllables(): ResultadosA['Syllables'] = TCM.SyllableAve(Samples)
    
    #Tree Amostras
    def TTreeA(): ResultadosA['Tree'] = Tree.DepthAveA(Samples)
    
    #SA Amostras
    def TSentimentA(): ResultadosA['Sentiment'] = SA_Spacy.SentimentA(Samples)
    
    #WR
    def TWordRarity(): ResultadosA['WR'] = WR.WordRarity(Amostras=Samples)
    
    #TR
    def TTR(): ResultadosA['TR'] = TR.TextTopic(Samples)
    
    #Emotions
    def TEmotions(): ResultadosA['Emotions'] = Emotions.Emotions(Samples)
    
    # Create threads
    threads = [
        threading.Thread(target=TSentenceLengthA),
        threading.Thread(target=TWordLengthA),
        threading.Thread(target=TLexicalDensityA),
        threading.Thread(target=TLexicalDiversityA),
        threading.Thread(target=TTreeA),
        threading.Thread(target=TSentimentA),
        threading.Thread(target=TReadA),
        threading.Thread(target=TSMOGA),
        threading.Thread(target=TTR),
        threading.Thread(target=TEmotions),
        threading.Thread(target=TWordRarity),
        threading.Thread(target=TSyllables),
    ]

    for thread in threads: thread.start()  
    for thread in threads: thread.join()

    return ResultadosA

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
            'LexicalDensity',
            'LexicalDiversity',
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
                ResultadosA['ARI'][1], 
                ResultadosA['Coleman'][1], 
                ResultadosA['Grade'][1], 
                ResultadosA['LexicalDensity'][0],
                ResultadosA['LexicalDiversity'][0],
                ResultadosA['SentenceLength'][1],
                ResultadosA['Sentiment'][0],
                ResultadosA['Sentiment'][1],
                ResultadosA['Sentiment'][2],
                ResultadosA['Sentiment'][3],
                ResultadosA['Smog'][1],
                ResultadosA['Tree'][1],
                ResultadosA['WordLength'][1],
                0,
                0
            ])

def Heuristics(File):
    df = pd.read_csv(File)
    
    dfs_num = df.drop(columns=['Text','ARI','Coleman','Grade','LexicalDensity','LexicalDiversity','SentenceLength','Smog','Tree','WordLength','Classification','ClassificationS'])
    
    df_num = df.drop(columns=['Text','Classification','SentimentNeu','ClassificationS','Compound','SentimentPos','SentimentNeg'])
    df_num = df_num.apply(pd.to_numeric, errors='coerce')
    df['Average'] = df_num.mean(axis=1).round(2)
    
    #Normalização
    conditions = [
        df['Average'] < 0.25,
        (df['Average'] >= 0.25) & (df['Average'] < 0.5),
        (df['Average'] >= 0.5) & (df['Average'] < 0.75),
        df['Average'] >= 0.75
    ]
    choices = [0, 1, 2, 3]
        
    conditionss = [
        (dfs_num['Compound'] < -0.05),
        (dfs_num['Compound'] >= -0.05) & (dfs_num['Compound'] <= 0.05),
        (dfs_num['Compound'] > 0.05)
    ]  
    #0: Negativo, 1:Neutro, 2:Positivo
    choicess = [0, 1, 2]
    
    df['Classification'] = np.select(conditions, choices)
    df['ClassificationS'] = np.select(conditionss, choicess)
    df.to_csv('B9.csv', index=False)
    
    #Dropa uma coluna especifica
    df = pd.read_csv('B9.csv')
    df = df.drop(columns=['Average'])
    df.to_csv('DataV9B_Spacy.csv', index=False) 

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

def ResultadosC(Samples):
    ResultadosA = {}
            
    def Read(): ResultadosA['Read'] = RM.Read(Samples)
     
    def TSentenceLengthA(): ResultadosA['SentenceLength'] = TCM.SentenceLength(Samples)
    def TWordLengthA(): ResultadosA['WordLength'] = TCM.WordLength(Samples)
    def TLexicalDensityA(): ResultadosA['LexicalDensity'] = TCM.LexicalDensity(Samples)
    def TLexicalDiversityA(): ResultadosA['LexicalDiversity'] = TCM.LexicalDiversity(Samples)
    
    def TTreeA(): ResultadosA['Tree'] = Tree.DepthAve(Samples)
    
    def TSentimentA(): ResultadosA['Sentiment'] = SA_Spacy.Sentiment(Samples)
    
    threads = [
        threading.Thread(target=TSentenceLengthA),
        threading.Thread(target=TWordLengthA),
        threading.Thread(target=TLexicalDensityA),
        threading.Thread(target=TLexicalDiversityA),
        threading.Thread(target=TTreeA),
        threading.Thread(target=TSentimentA),
        threading.Thread(target=Read),
    ]
    for thread in threads: thread.daemon = True
    for thread in threads: thread.start()
        
    for thread in threads: thread.join()

    return ResultadosA

def CSV_Column(input_file, column_name, output_file, word_limit=5000):
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
        
        def has_more_than_word_limit(text, limit):
            if isinstance(text, str):
                return len(text.split()) > limit
            return False

        if column_name in df.columns:
            df_filtered = df[df[column_name].apply(lambda x: has_more_than_word_limit(x, word_limit))]
            df_filtered.to_csv(output_file, index=False)  
    except Exception as e:
        print(f"An error occurred: {e}")

#CSV_Column("Articles.csv", "Article", "ArticlesFiltred.csv")
#txt_to_csv("D:/GitHub Repository/Projeto/Projeto_Versão_Final/Samples/ComunTexts/Simple(LVL6)","NovelTextSimple.csv")