#Grammar Checking
import requests

def File(File):
    with open(File, "r", encoding='utf-8', errors='ignore') as file:
        sentence = file.read().replace("\n", " ")
    return sentence

def check_grammar(text):
    api_url = "https://languagetool.org/api/v2/check"
    
    params = {
        "text": text,
        "language": "en-US",
    }
    
    try:
        response = requests.post(api_url, data=params)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()

        if 'matches' in data:
            for match in data['matches']:
                print(f"Type: {match['rule']['description']}")
                print(f"Message: {match['message']}")
                print(f"Context: {match['context']['text']}")
                
                if match['replacements']:
                    print(f"Suggested Correction: {match['replacements'][0]['value']}")
                else:
                    print("No suggested correction available.")
                    
                print()
        else:
            print("No grammatical errors found.")
    except requests.RequestException as e:
        print(f"Error: {e}")

text = File("Textos/obama.txt")

check_grammar(text)
