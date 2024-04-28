import spacy, random, csv, Tree, TCM, os, threading, RM, SA, GC, sys

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 1600000

maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

def File(File):
    with open(File, "r", encoding='utf-8', errors='ignore') as file:
        sentence = file.read()
    return sentence

def Amostras(Texto):
    doc = nlp(Texto)

    Palavras = [token.text for token in doc if token.is_alpha] # Separa cada token, guardando só palavras 
    
    paragraphs = []
    current_paragraph = ""
    sentences_count = 0

    # Separa cada paragrafo 
    for sent in doc.sents:
        current_paragraph += sent.text + " "
        sentences_count += 1
        if sentences_count == 10:
            paragraphs.append(current_paragraph.strip())
            current_paragraph = ""
            sentences_count = 0

    # Add the last paragraph if it's not complete
    if current_paragraph:
        paragraphs.append(current_paragraph.strip())
    
    Paragrafos = random.sample(paragraphs,10) 

    #Lista onde é guardada 100 repetições com amostras de 100 palavras da lista Palavras
    Samples = []
    for _ in range(100):
        Sample = random.sample(Palavras,100)
        Samples.append(Sample)
    
    return Samples, Paragrafos # [0] Escolhe de forma random 100 amostras de 100 palavras de uma lista tokenizada; [1] Escolhe de forma random 10 paragrafos de um texto

def txt_csv(input_dir, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
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
            'Sentiment',
            'Smog',
            'Tree',
            'WordLength'
            ])

        for filename in os.listdir(input_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as txtfile:
                    Texto = txtfile.read().replace("\n\n", "  ").replace("\n", " ")
                    Samples = Amostras(Texto)
                    
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
                    def TGrammar(): ResultadosA['Grammar'] = GC.Grammar(Texto)

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
                    for thread in threads:
                        thread.start()

                    # Wait for all threads to finish
                    for thread in threads:
                        thread.join()

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
                        ResultadosA['Sentiment'],
                        ResultadosA['Smog'][0],
                        ResultadosA['Tree'][0],
                        ResultadosA['WordLength'][0]
                    ]) 

def count_words(text):
    return len(text.split())

def filter_and_write_texts(input_csv, output_csv):
    filtered_texts = []

    with open(input_csv, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            text = row['text']
            if count_words(text) > 10000:
                filtered_texts.append(text)

    with open(output_csv, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['text'])

        for text in filtered_texts:
            writer.writerow([text])

#txt_csv("Textos/","teste.csv")
filter_and_write_texts("blogtext.csv", "teste2.csv")