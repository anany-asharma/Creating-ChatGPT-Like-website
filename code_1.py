from flask import Flask, request, render_template
from flask_cors import CORS
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# download a template chatbot webpage and configure it to make requests to your backend server.
app=Flask(__name__)
CORS(app)

# we will be using "facebook/blenderbot-400M-distill" model as t has an open source liense and works relatively fast

model_name="facebook/blenderbot-400M-distill"

# AutoModelForSeq2SeqLM allows us to interact with the chosen model
model=AutoModelForSeq2SeqLM.from_pretrained(model_name)
# AutoTokenizer converts our input into tokens and passes it to language mode efficiently
tokenizer=AutoTokenizer.from_pretrained(model_name)

# keeping track of conversation history is important when interacting with a chatbot because as chatbot refers previous conversation when generating output.
conversation_history=[]

@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/chatbot',methods=['POST'])
def handle_prompt():
    # prompting user for input
    data=request.get_data(as_text=True)
    data=json.loads(data)
    print(data)
    input_text=data['prompt']

    # during each interaction we will be passing conversation history with input to the model
    history='\n'.join(conversation_history)

    # tokenization of user prompt and chat history
    inputs=tokenizer.encode_plus(history,input_text,return_tensors="pt")

    # generate outputs from the model
    outputs=model.generate(**inputs)

    # decoding output
    response=tokenizer.decode(outputs[0],skip_special_tokens=True).strip()

    # updating conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)

    return response

if __name__=="__main__":
    app.run(debug=True)
