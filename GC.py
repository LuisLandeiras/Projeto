#Grammar Checking
import language_tool_python, AuxFun

def Grammar(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    
    print(len(matches))
    
    #match atributtes:
    #ruleId,message,replacements,offsetInContext,context,offset,errorLength,category,ruleIssueType,sentence
    for match in matches:
        print("Error:", match.message)
        print("Error Type:", match.ruleIssueType)
        print("Sentence:", match.sentence)
        print("Context:", match.context)
        print("Possible Replacements:", match.replacements)
        print("Possible Correction:", tool.correct(match.sentence))
        print("------------------------------------")
    
text = AuxFun.File("Textos/obama.txt")

Grammar(text)