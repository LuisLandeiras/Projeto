from flask import Flask, render_template, request
import AuxFun, TCM, RM, SA, GC, Tree, threading, time

app = Flask(__name__)

@app.route("/")
def main():
    return render_template('index.html')

@app.route('/sub', methods=['POST'])
def sub():
    t = time.process_time()
    result = request.form
    Texto = result.get('text')

    ResultadosA = {}
    Resultados = {}

    Samples = AuxFun.Amostras(Texto)
    
    #RM Amostras
    def TSMOGA(): ResultadosA['Smog'] = RM.SMOGA(Samples[1])
    def TColemanA(): ResultadosA['Coleman'] = RM.ColemanA(Samples[1])
    def TGradeA(): ResultadosA['Grade'] = RM.FleschGradeA(Samples[1])
    def TReadingA(): ResultadosA['Reading'] = RM.FleschReadingA(Samples[1])
    def TARIA(): ResultadosA['ARI'] = RM.ARIA(Samples[1])
    #RM
    def TRead(): Resultados['Read'] = RM.Read(Texto)
    
    #TCM Amostras
    def TSentenceLengthA(): ResultadosA['SentenceLength'] = TCM.SentenceLengthA(Samples[1])
    def TWordLengthA(): ResultadosA['WordLength'] = TCM.WordLengthA(Samples[0])
    def TLexicalDensityA(): ResultadosA['LexicalDensity'] = TCM.LexicalDensityA(Samples[0])
    def TLexicalDiversityA(): ResultadosA['LexicalDiversity'] = TCM.LexicalDiversityA(Samples[0])
    #TCM 
    def TSentenceLength(): Resultados['SentenceLength'] = TCM.SentenceLength(Texto)
    def TWordLength(): Resultados['WordLength'] = TCM.WordLength(Texto)
    def TLexicalDensity(): Resultados['LexicalDensity'] = TCM.LexicalDensity(Texto)
    def TLexicalDiversity(): Resultados['LexicalDiversity'] = TCM.LexicalDiversity(Texto)
    
    #Tree Amostras
    def TTreeA(): ResultadosA['Tree'] = Tree.DepthAveA(Samples[1])
    #Tree
    def TTree(): Resultados['Tree'] = Tree.DepthAve(Texto)
    
    #SA Amostras
    def TSentimentA(): ResultadosA['Sentiment'] = SA.SentimentA(Samples[1])
    #SA
    def TSentiment(): Resultados['Sentiment'] = SA.Sentiment(Texto)
    
    #GC
    #def TGrammar(): Resultados['Grammar'] = GC.Grammar(Texto)
    
    # Create threads
    threads = [
        threading.Thread(target=TSentenceLengthA),
        threading.Thread(target=TWordLengthA),
        threading.Thread(target=TLexicalDensityA),
        threading.Thread(target=TLexicalDiversityA),
        threading.Thread(target=TTreeA),
        threading.Thread(target=TSentimentA),
        threading.Thread(target=TSentenceLength),
        threading.Thread(target=TWordLength),
        threading.Thread(target=TLexicalDensity),
        threading.Thread(target=TLexicalDiversity),
        threading.Thread(target=TTree),
        threading.Thread(target=TSentiment),
        #threading.Thread(target=TGrammar),
        threading.Thread(target=TGradeA),
        threading.Thread(target=TSMOGA),
        threading.Thread(target=TColemanA),
        threading.Thread(target=TReadingA),
        threading.Thread(target=TARIA),
        threading.Thread(target=TRead)
    ]
    
    # Start threads
    for thread in threads:
        thread.start()
    
    # Wait for all threads to finish
    for thread in threads:
        thread.join()
    
    return render_template('sub.html', Amostras=ResultadosA, Completo=Resultados, Time=time.process_time() - t)
