#Grammar Checking
import language_tool_python, AuxFun, time

def Grammar(text):
    t = time.process_time()
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    
    Resultados = []
    
    #match atributtes:
    #ruleId,message,replacements,offsetInContext,context,offset,errorLength,category,ruleIssueType,sentence
    for match in matches:
        Resultados.append(match)

    return Resultados, time.process_time() - t
#text = AuxFun.File("Textos/obama.txt")

#Grammar(text)