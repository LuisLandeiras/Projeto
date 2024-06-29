#Grammar Checking
import language_tool_python, time

def Grammar(Samples):
    t = time.process_time()
    tool = language_tool_python.LanguageTool('en-US')
    Soma = 0
    for Sample in Samples:
        matches = tool.check(Sample)
        Soma += len(matches)
    
    #Resultados = []
    
    #match atributtes:
    #ruleId,message,replacements,offsetInContext,context,offset,errorLength,category,ruleIssueType,sentence
    #for match in matches:
    #    Resultados.append(match)

    return Soma, time.process_time() - t