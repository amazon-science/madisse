from bs4 import BeautifulSoup

def parse_output_response(response):
    soup = BeautifulSoup(response, 'html.parser')
    explanation_list = soup.findAll("explanation")
    explanation_text = ""
    for exp in explanation_list:
        if exp.string != None:
            explanation_text += exp.string + ' '
        else:
            explanation_text = response
    explanation_text = ' '.join(explanation_text.split())
    if len(soup.findAll("label")) > 0:
        labels = soup.findAll("label")[-1].string.strip()
    else:
        labels = "Unknown"
    return labels, explanation_text

def parse_output_response_w_category(response):
    soup = BeautifulSoup(response, 'html.parser')
    explanation_list = soup.findAll("explanation")
    explanation_text = ""
    for exp in explanation_list:
        if exp.string != None:
            explanation_text += exp.string + ' '
        else:
            explanation_text = response
    explanation_text = ' '.join(explanation_text.split())
    
    category_list = soup.findAll("category")
    category_text = ""
    for exp in category_list:
        if exp.string != None:
            category_text += exp.string + ' '
        else:
            category_text = ""
    category_text = ' '.join(category_text.split())

    if len(soup.findAll("label")) > 0:
        labels = soup.findAll("label")[-1].string.strip()
    else:
        labels = "Unknown"

    return labels, category_text, explanation_text

def parse_output_w_chat_label(response):
    soup = BeautifulSoup(response, 'html.parser')
    argument_list = soup.findAll("argument")
    argument_text = ""
    for argument in argument_list:
        if argument.string != None:
            argument_text += argument.string + ' '
        else:
            argument_text = response
    argument_text = ' '.join(argument_text.split())
    if len(soup.findAll("label")) > 0:
        guidelines = soup.findAll("label")[0].string.strip()
    else:
        guidelines = "Unknown"
    
    return argument_text, guidelines
